from statemachine import StateMachine, State
import time


class MyStateMachine(StateMachine):
        
    State_1 = State(initial=True)
    State_2 = State()
    
    Trans_1 = (
        State_1.to(State_2) |
        State_2.to(State_1)
    )
    
    def __init__(self):       
        super(MyStateMachine, self).__init__()

    def on_enter_State_1(self):
        print("Entering State_1 state")               
        self.long_running_task()
      
    def on_exit_State_1(self):
        print("Exiting State_1 state")

    def on_enter_State_2(self):
        print("Entering State_2 state")               
        self.long_running_task()
      
    def on_exit_State_2(self):
        print("Exiting State_2 state")
       
    
    def long_running_task(self):
        print("long running task process started")
        time.sleep(3)
        self.Trans_1()  
        print("long running task process ended")      
  
###############################################################################

sm = MyStateMachine() # state machine started just by creating an instance

try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print('\ninterrupted!')
