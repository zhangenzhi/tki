
from .cifar10_rl_student import Cifar10RLStudent
from .cifar100_rl_student import Cifar100RLStudent
from .mnist_rl_student import MnistRLStudent

from .cifar10_rl_supervisor import Cifar10RLSupervisor
from .cifar100_rl_supervisor import Cifar100RLSupervisor
from .mnist_rl_supervisor import MnistRLSupervisor


class StudentFactory():
    def __init__(self) -> None:
        self.student_list = { 'cifar10': Cifar10RLStudent, 'cifar100':Cifar100RLStudent,
                             'mnist':MnistRLStudent}

    def __call__(self, student_args, supervisor = None, id = 0):
        return self.get_student(student_args=student_args, 
                           supervisor=supervisor, 
                           id=id)

    def get_student(self, student_args, supervisor = None, id = 0):
        student_cls = self.student_list.get(student_args['dataloader']['name'])
        
        return student_cls(student_args=student_args, 
                           supervisor=supervisor, 
                           id=id)


class SupervisorFactory():
    def __init__(self) -> None:
        self.supervisor_list = {'cifar10': Cifar10RLSupervisor, 'cifar100':Cifar100RLSupervisor,
                             'mnist':MnistRLSupervisor}
    
    def __call__(self, supervisor_args, student_target='', id = 0):
        return self.get_supervisor(supervisor_args=supervisor_args, student_target=student_target, id=id)

    def get_supervisor(self, supervisor_args = None, student_target='', id = id):
        supervisor_cls = self.supervisor_list.get(student_target['name'])
        print(supervisor_cls)
        return supervisor_cls(supervisor_args=supervisor_args, id = id)

student_factory = StudentFactory()
supervisor_factory = SupervisorFactory()