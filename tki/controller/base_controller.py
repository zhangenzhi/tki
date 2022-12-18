import os

import tensorflow as tf

from tki.tools.utils import *
from tki.dataloader.utils import *
from tki.controller.utils import *
from tki.train.factory import student_factory, supervisor_factory


class BaseController(object):
    def __init__(self, yaml_configs):
    
        self.yaml_configs = yaml_configs

        self._build_enviroment()
        weight_dir = os.path.join(self.log_path, "weight_space")
        if os.path.exists(weight_dir):
            self._student_ids = len(glob_tfrecords(weight_dir, glob_pattern='*.tfrecords'))
        else:
            self._student_ids = 0
        self._supervisor_ids = 0
        
        self.supervisor = self._build_supervisor()


    def _build_enviroment(self):
        self.args = self.yaml_configs['experiment']
        self.context = self.args['context']
        # self.devices = list(range(self.context['devices']))
        self.log_path = os.path.join(self.context['log_path'], self.context['name'])
        

    def _build_student(self, supervisor=None):
        student_args = self.args["student"]
        student_args['context'] = self.context
        student_args['log_path'] = self.log_path

        student = student_factory(student_args=student_args, 
                                  supervisor = supervisor,
                                  id = self._student_ids)

        self._student_ids += 1
        return student

    def _build_supervisor(self):
        supervisor_args = self.args["supervisor"]
        supervisor_args['context'] = self.context
        supervisor_args['log_path'] = self.log_path

        supervisor = supervisor_factory(supervisor_args=supervisor_args,
                                        id = self._supervisor_ids)

        self._supervisor_ids += 1
        return supervisor
        
    def warmup(self, warmup):
        init_samples = warmup['student_nums']
        supervisor_iters = warmup['supervisor_iters']
        
        for i in range(init_samples):
            student = self._build_student(supervisor=self.supervisor)
            student.run()
        
        for j in range(supervisor_iters):
            keep_train = False if j == 0 else True
            self.supervisor.run(keep_train=keep_train, new_students=[])

    def main_loop(self):

        main_loop = self.args['main_loop']

        # init weights pool
        if 'warmup' in main_loop:
            self.warmup(main_loop['warmup'])

        # main loop
        for j in range(main_loop['nums']):
            new_students = []
            for i in range(main_loop['student_nums']):
                student = self._build_student(supervisor=self.supervisor)
                new_students.append(student.run())
                    
            # supervisor
            print_green("new_student:{}, welcome!".format(new_students))
            self.supervisor.run(keep_train=True, new_students=new_students)

    def run(self):
        print_green("Start to run!")
        
        self.main_loop()

        print_green('[Task Status]: Task done! Time cost: {:}')

    

