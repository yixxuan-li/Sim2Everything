from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from inspire_sdkpy import inspire_dds, inspire_hand_defaut

import numpy as np
from enum import IntEnum
import threading
import time
from multiprocessing import Process, Array

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

Inspire_Num_Motors = 6
kTopicLeftInspireCommand = "rt/inspire_hand/ctrl/l"
kTopicRightInspireCommand = "rt/inspire_hand/ctrl/r"
kTopicLeftInspireState = "rt/inspire_hand/state/l"
kTopicRightInspireState = "rt/inspire_hand/state/r"

class InspireController:
    def __init__(self, left_q_target, right_q_target, fps = 100.0):
        logger_mp.info("Initialize InspireController...")
        self.fps = fps
        ChannelFactoryInitialize(0)

        # initialize handcmd publisher and handstate subscriber
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicLeftInspireCommand, inspire_dds.inspire_hand_ctrl)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicRightInspireCommand, inspire_dds.inspire_hand_ctrl)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicLeftInspireState, inspire_dds.inspire_hand_state)
        self.LeftHandState_subscriber.Init(self._subscribe_hand_state_left, 10)
        self.RightHandState_subscriber = ChannelSubscriber(kTopicRightInspireState, inspire_dds.inspire_hand_state)
        self.RightHandState_subscriber.Init(self._subscribe_hand_state_right, 10)

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', Inspire_Num_Motors, lock=True)  
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)

        while True:
            if any(self.right_hand_state_array) and any(self.left_hand_state_array): # any(self.left_hand_state_array) and 
                break
            time.sleep(0.01)
            logger_mp.warning("[InspireController] Waiting to subscribe dds...")
        logger_mp.info("[InspireController] Subscribe dds ok.")

        self.left_q_target_array = Array('i', Inspire_Num_Motors, lock=True)
        self.right_q_target_array = Array('i', Inspire_Num_Motors, lock=True)
        hand_control_process = Process(target=self._control_process, args=(self.left_q_target_array, self.right_q_target_array))
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize InspireController OK!\n")

    def _subscribe_hand_state_left(self, msg: inspire_dds.inspire_hand_state):
        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
            self.left_hand_state_array[idx] = msg.pos_act[id]

    def _subscribe_hand_state_right(self, msg: inspire_dds.inspire_hand_state):
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
            self.right_hand_state_array[idx] = msg.pos_act[id]

    def _ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        Set current left, right hand motor state target q
        """
        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):             
            self.left_hand_msg.angle_set[idx] = left_q_target[idx]         
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):             
            self.right_hand_msg.angle_set[idx] = right_q_target[idx] 

        self.LeftHandCmb_publisher.Write(self.left_hand_msg)
        self.RightHandCmb_publisher.Write(self.right_hand_msg)
        # logger_mp.debug("hand ctrl publish ok.")
    
    def _control_process(self, left_q_target_array, right_q_target_array):
        self.running = True

        # initialize inspire hand's cmd msg
        self.left_hand_msg = inspire_hand_defaut.get_inspire_hand_ctrl()
        self.right_hand_msg = inspire_hand_defaut.get_inspire_hand_ctrl()
        self.left_hand_msg.mode = 0b0001
        self.right_hand_msg.mode = 0b0001

        try:
            while self.running:
                start_time = time.time()
                with left_q_target_array.get_lock():
                    l_q_target = [left_q_target_array[i] for i in range(Inspire_Num_Motors)]
                with right_q_target_array.get_lock():
                    r_q_target = [right_q_target_array[i] for i in range(Inspire_Num_Motors)]
                self._ctrl_dual_hand(l_q_target, r_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller has been closed.")

    def set_q_target(self, left_q_target, right_q_target):
        with self.left_q_target_array.get_lock():
            self.left_q_target_array[:] = np.clip(int(left_q_target * 1000), 0, 1000)
        with self.right_q_target_array.get_lock():
            self.right_q_target_array[:] = np.clip(int(right_q_target * 1000), 0, 1000)

# Update hand state, according to the official documentation, https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand
# the state sequence is as shown in the table below
# ┌──────┬───────┬──────┬────────┬────────┬────────────┬────────────────┬───────┬──────┬────────┬────────┬────────────┬────────────────┐
# │ Id   │   0   │  1   │   2    │   3    │     4      │       5        │   6   │  7   │   8    │   9    │    10      │       11       │
# ├──────┼───────┼──────┼────────┼────────┼────────────┼────────────────┼───────┼──────┼────────┼────────┼────────────┼────────────────┤
# │      │                    Right Hand                                │                   Left Hand                                  │
# │Joint │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │
# └──────┴───────┴──────┴────────┴────────┴────────────┴────────────────┴───────┴──────┴────────┴────────┴────────────┴────────────────┘
class Inspire_Right_Hand_JointIndex(IntEnum):
    kRightHandPinky = 0
    kRightHandRing = 1
    kRightHandMiddle = 2
    kRightHandIndex = 3
    kRightHandThumbBend = 4
    kRightHandThumbRotation = 5

class Inspire_Left_Hand_JointIndex(IntEnum):
    kLeftHandPinky = 0
    kLeftHandRing = 1
    kLeftHandMiddle = 2
    kLeftHandIndex = 3
    kLeftHandThumbBend = 4
    kLeftHandThumbRotation = 5