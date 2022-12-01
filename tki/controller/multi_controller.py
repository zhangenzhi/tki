import os
import time
import tensorflow as tf
import multiprocessing as mp
from multiprocessing import Pool, Queue, Process

from tki.tools.utils import *
from tki.train.supervisor import Supervisor
from tki.train.utils import ForkedPdb
from tki.controller.utils import *
from tki.controller.base_controller import BaseController


class MultiController(BaseController):
    def __init__(self, yaml_configs):
        super(MultiController, self).__init__(yaml_configs=yaml_configs)
        self.queue = Queue(maxsize=100)
        self.sp_queue = Queue(maxsize=100)
        self.parallel = yaml_configs['experiment']['context']['multi-p']

    def _build_enviroment(self):
        mp.set_start_method("spawn")
            
        self.args = self.yaml_configs['experiment']
        self.context = self.args['context']
        self.devices = list(range(self.context['devices']))
        self.log_path = os.path.join(self.context['log_path'], self.context['name'])

    def device_dispatch(self, student):
        num_gpus = len(self.devices)
        avail_gpu = student.id % num_gpus
        return self.devices[avail_gpu]
        
    def warmup(self, warmup):
        init_samples = warmup['student_nums']
        supervisor_iters = warmup['supervisor_iters']
        
        processes = []
        new_students = []
        for i in range(init_samples):
            student = self._build_student()
            devices = str(self.device_dispatch(student=student))
            p = StudentProcess(student=student, connect_queue=self.queue, supervisor_info=None, devices=devices)
            p.start()
            processes.append(p)
            time.sleep(2)
            if (i+1) % self.parallel == 0:
                pres = [p.join() for p in processes]
                processes = []
        while not self.queue.empty():
            new_students += [self.queue.get()]
   
        for j in range(supervisor_iters):
            keep_train = False if j == 0 else True
            self.supervisor.run(keep_train=keep_train, new_students=[])
            
    def main_loop(self):
        main_loop = self.args['main_loop']

        # init weights pool
        if 'warmup' in main_loop:
            self.warmup(main_loop['warmup'])

        # main loop
        total_students = [self._build_student() for i in range(main_loop['student_nums']*main_loop['nums'])]
        for j in range(main_loop['nums']):
            # mp students with supervisor
            processes = []
            print_green(total_students[0])
            for i in range(main_loop['student_nums']):
                student = total_students.pop(0)
                devices = str(self.device_dispatch(student=student))
                supervisor_info = {"logdir": self.supervisor.logdir}
                p = StudentProcess(student=student, connect_queue=self.queue, supervisor_info=supervisor_info, devices=devices)

                p.start()
                processes.append(p)
                time.sleep(3)
                
            pres = [p.join() for p in processes]
            new_students = []
            while not self.queue.empty():
                new_students += [self.queue.get()]
                    
            # supervisor
            print_green("new_student:{}, welcome!".format(new_students))
            self.supervisor.run(keep_train=True, new_students=new_students)


class StudentProcess(mp.Process):
    def __init__(self, student, connect_queue=None, supervisor_info=None, devices='0'):
        super().__init__()
        print_green("Init Student:{} Process on Device:{}.".format(student.id, devices))

        os.environ['CUDA_VISIBLE_DEVICES'] = devices

        self.student = student 
        self.connect_queue = connect_queue
        self.supervisor_info = supervisor_info
        self.devices= devices
        return

    def run(self):

        gpus = tf.config.experimental.list_physical_devices("GPU")
        print_red("Init Student:{} Process on Device:{}.".format(self.student.id, gpus))
        self.student.run(connect_queue=self.connect_queue, supervisor_info=self.supervisor_info, devices=self.devices)


class SupervisorProcess(mp.Process):
    def __init__(self, supervisor, keep_train, new_students, queue, devices='0'):
        super().__init__()
        print_green("Init Supervisor:{} Process on Device:{}.".format(supervisor.id, devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = devices
        self.supervisor = supervisor 
        self.keep_train = keep_train
        self.new_students = new_students
        self.devices= devices
        self.queue = queue
        return

    def run(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.devices
        self.gpus = tf.config.experimental.list_physical_devices("GPU")
        print(self.gpus)
        self.supervisor.run(keep_train=self.keep_train, new_students=[])
        print("put supervisor")
        ForkedPdb().set_trace()
        self.queue.put(self.supervisor)
        