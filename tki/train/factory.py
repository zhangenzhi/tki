
from .naive_student import NaiveStudent

from .naive_supervisor import NaiveSupervisor


class StudentFactory():
    def __init__(self) -> None:
        self.student_list = {'naive':NaiveStudent}

    def __call__(self, student_args, supervisor = None, id = 0):
        return self.get_student(student_args=student_args, 
                           supervisor=supervisor, 
                           id=id)

    def get_student(self, student_args, supervisor = None, id = 0):
        student_cls = self.student_list.get(student_args['name'])
        
        return student_cls(student_args=student_args, 
                           supervisor=supervisor, 
                           id=id)


class SupervisorFactory():
    def __init__(self) -> None:
        self.supervisor_list = {'naive': NaiveSupervisor}
    
    def __call__(self, supervisor_args, student_target='', id = 0):
        return self.get_supervisor(supervisor_args=supervisor_args, id=id)

    def get_supervisor(self, supervisor_args = None, id = id):
        supervisor_cls = self.supervisor_list.get(supervisor_args['name'])
        print(supervisor_cls)
        return supervisor_cls(supervisor_args=supervisor_args, id = id)

student_factory = StudentFactory()
supervisor_factory = SupervisorFactory()