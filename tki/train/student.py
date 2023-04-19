import os
from tqdm import trange
import tensorflow as tf
from easydict import EasyDict as edict

# others
from tki.train.modules.RL.action import ActionSpace
from tki.train.modules.RL.policy import PolicySpace
from tki.train.modules.RL.replay_buffer import ReplayBuffer

from tki.train.trainer import Trainer


class Student(Trainer):
    def __init__(self, student_args, supervisor=None, supervisor_info=None, id=0):
        super(Student, self).__init__(trainer_args=student_args, id=id)
        
        ## RL
        self.action = 1.0
        self.c_flag = False
        self.act_space = ActionSpace(action_args=student_args.train_loop.train.action)
        self.policy = PolicySpace(policy_args=student_args.train_loop.train.policy)
        self.training_knowledge = ReplayBuffer(buffer_args=student_args.train_loop.valid, 
                                               log_path=student_args.log_path,
                                               student_id=self.id)
        
        # load supervisor
        self.supervisor = None
        if supervisor != None:
            self.supervisor = supervisor
        else:
            self.supervisor = self._load_supervisor_model(supervisor_info)

    def _load_supervisor_model(self, supervisor_info, model_name="DNN_latest"):
        if self.supervisor == None:
            if supervisor_info == None:
                return None
            else:
                sp_model_logdir = os.path.join(supervisor_info["logdir"], model_name)
                supervisor = tf.keras.models.load_model(sp_model_logdir)
        else:
            supervisor = self.supervisor
        return supervisor

    def update_supervisor(self, inputs, labels):
        
        sp_opt = tf.keras.optimizers.SGD(0.01)
        sp_loss_fn = tf.keras.losses.MeanSquaredError()
        
        with tf.GradientTape() as tape:
            predictions = self.supervisor(inputs)
            loss = sp_loss_fn(labels, predictions)
            gradients = tape.gradient(
                loss, self.supervisor.trainable_variables)
            sp_opt.apply_gradients(
                zip(gradients, self.supervisor.trainable_variables))
            
        return loss
    
    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels, first_batch=False, action=1.0):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(labels, predictions)
            train_metrics = tf.reduce_mean(self.train_metrics(labels, predictions))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            if not first_batch:
                grads = [action*g for g in gradients]
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.mt_loss_fn.update_state(loss)
        
        return loss, gradients, train_metrics
    
    def train_block(self, epoch, train_steps_per_epoch, train_iter, valid_args, valid_iter):
        with trange(train_steps_per_epoch, desc="Train steps", leave=False) as t:
            self.mt_loss_fn.reset_states()
            self.train_metrics.reset_states()
            for train_step in t:
                
                # train
                data = train_iter.get_next()
                first_batch = True if epoch*train_steps_per_epoch+train_step == 0 else False
                train_loss, train_gard, train_metrics = self._train_step(data['inputs'], data['labels'], 
                                                                         first_batch=first_batch, action=self.action)
                
                
                # valid
                if (epoch*train_steps_per_epoch+train_step) % valid_args['valid_gap'] == 0:
                    
                    expect_q_values, state = self.supervisor(self.model.trainable_variables)
                    act_idx = self.policy(expect_q_values, self.id)
                    action = self.act_space(act_idx)
                    self.action = action
                    self.c_flag = True
                    
                    valid_loss, valid_metrics = self.valid_block(train_step, valid_args, valid_iter)
                    
                    # training knowledge
                    q_value = self.training_knowledge.update_buffer(state=state,
                                                          expect_q_values=expect_q_values, 
                                                          train_loss=train_loss,
                                                          train_metric=train_metrics,
                                                          act_grad = train_gard,
                                                          action=action,
                                                          act_idx=act_idx,
                                                          valid_loss=valid_loss, 
                                                          valid_metric=valid_metrics,
                                                          step=epoch*train_steps_per_epoch+train_step)
                    
                    with self.logger.as_default():
                        tf.summary.scalar("q_value", q_value, step=epoch*train_steps_per_epoch+train_step)
                        tf.summary.scalar("lr", self.optimizer.lr, step=epoch*train_steps_per_epoch+train_step)
                        if self.act_space.act_style == "2_dims":
                            tf.summary.scalar("act_lr",  action[0], step=epoch*train_steps_per_epoch+train_step)
                            tf.summary.scalar("act_re",  action[1], step=epoch*train_steps_per_epoch+train_step)
                        else:
                            tf.summary.scalar("action",  action, step=epoch*train_steps_per_epoch+train_step)
                            
            etr_loss = self.mt_loss_fn.result()
            etr_metric = self.train_metrics.result()
                    
            return etr_loss, etr_metric

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _tki_train_step(self, inputs, labels, action):
        raise NotImplementedError("trainer need _tki_train_step func.")

    def run(self, connect_queue=None, devices='1'):

        # set enviroment
        # self._build_enviroment()

        # prepare dataset
        self.train_dataset, self.valid_dataset, self.test_dataset, \
            self.dataloader = self._build_dataset()

        # build optimizer
        self.optimizer = self._build_optimizer()

        # build model
        self.model = self._build_model()

        # build losses and metrics
        self.loss_fn, self.mt_loss_fn, self.mv_loss_fn, self.mtt_loss_fn = self._build_loss_fn()
        self.train_metrics, self.valid_metrics, self.test_metrics = self._build_metrics()
        
        # build weights save writter
        self.logger = self._build_logger()
   
        self.train()
        
        reward = self.training_knowledge.save_experience()
        with self.logger.as_default():
            for idx in range(len(reward)):
                tf.summary.scalar("reward", reward[idx], step=idx)
        
        print('Finished training student {}'.format(self.id))

        if connect_queue != None:
            connect_queue.put(self.training_knowledge.weight_file)

        return self.training_knowledge.weight_file
