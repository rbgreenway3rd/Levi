from statemachine import StateMachine, State
import time
import pydot
import threading
from typing import Optional
import os 
from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss
from eff_word_net import samples_loc



from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import pyaudio
from scipy.io.wavfile import write
import torch
import numpy as np

from audiotools import AudioPlayer, MicrophoneRecorder, write_wav


# constants
WAIT_FOR_VOICE_ON_TIMEOUT = 5.0 # seconds
WAIT_FOR_VOICE_OFF_TIMEOUT = 20.0 
WAIT_FOR_RESPONSE_TIMEOUT = 10.0
WAIT_UPDATE_DELAY = 0.1






class AssistantStateMachine(StateMachine):
    '''
    A state machine for Levi
    '''

    # define states
    HotWordDetect = State(initial=True)
    WaitForVoiceOn = State()
    WaitForVoiceOff = State()
    SendMessage = State()
    WaitForResponse = State()
    PlayResponse = State()
    Timeout = State()
    CommError = State()

    # define events that cause a transition
    event_HotWordDetected = HotWordDetect.to(WaitForVoiceOn)
    event_VoiceOnDetected = WaitForVoiceOn.to(WaitForVoiceOff)
    event_VoiceOffDetected = WaitForVoiceOff.to(SendMessage)
    event_MessageSent = SendMessage.to(WaitForResponse)
    event_ResponseReceived = WaitForResponse.to(PlayResponse)

    event_Timeout = (
        WaitForVoiceOn.to(Timeout)  |
        WaitForVoiceOff.to(Timeout) |
        WaitForResponse.to(Timeout)
        )
    
    event_CommError = SendMessage.to(CommError)

    event_Go = (
        Timeout.to(HotWordDetect) |
        CommError.to(HotWordDetect) |
        PlayResponse.to(HotWordDetect)
    )

    event_HotWordLoop = HotWordDetect.to(HotWordDetect)

    timeout_thread : threading.Thread
    worker_thread  : threading.Thread
    
    
    

    def __init__(self):
        
        self.done = False 
        self.timeout = False       
        self.recording = []
        self.audioplayer:AudioPlayer = None
        self.micrecorder:MicrophoneRecorder = MicrophoneRecorder()

        self.base_model = Resnet50_Arc_loss()

        self.levi_hw = HotwordDetector(
        hotword="levi",
        model = self.base_model,
        reference_file="/home/bryan/workspace/levi/levi_ref.json",
        threshold=0.7,
        relaxation_time=2
        )

        self.mic_stream = SimpleMicStream(
            window_length_secs=1.5,
            sliding_window_secs=0.75,
        )

        self.silero_model = load_silero_vad(onnx=True)
    
        super(AssistantStateMachine, self).__init__()
   

    ###########################################################
    # define Enter/Exit functions for each state

    def on_enter_HotWordDetect(self):
        # print("Entering HotWordDetect state")
                    
        hotword = self.listen_for_hotwords()
        if hotword is None:
            self.event_HotWordLoop() # restart hot word detect
        elif hotword == 'Levi':
            self.event_HotWordDetected()
        else:
            print(f"Unhandled hotword detected: {hotword}" )
            self.event_HotWordLoop()
        
    def on_exit_HotWordDetect(self):
        # print("Exiting HotWordDetect state")
        pass
      
       

    def on_enter_WaitForVoiceOn(self):
        # print("Entering WaitForVoiceOn state")
        self.start_timeout_timer(WAIT_FOR_VOICE_ON_TIMEOUT)
        self.start_recording()        
        self.listen_for_voice()
        self.stop_timeout_timer()
        if self.timeout:
            self.timeout = False
            self.stop_recording()
            self.event_Timeout()
        else: 
            self.event_VoiceOnDetected()
        
    def on_exit_WaitForVoiceOn(self):
        # print("Exiting WaitForVoiceOn state")
        pass



    def on_enter_WaitForVoiceOff(self):
        # print("Entering WaitForVoiceOff state")
        self.start_timeout_timer(WAIT_FOR_VOICE_OFF_TIMEOUT)
        self.listen_for_silence() 
        self.stop_timeout_timer()
        self.stop_recording()
        if self.timeout:
            self.timeout = False            
            self.event_Timeout()
        else:       
            self.event_VoiceOffDetected()

    def on_exit_WaitForVoiceOff(self):
        # print("Exiting WaitForVoiceOff state")
        pass
        


    def on_enter_SendMessage(self):
        # print("Entering SendMessage state")
        success = self.send_message()
        if success:
            self.event_MessageSent()
        else:
            self.event_CommError()
   
    def on_exit_SendMessage(self):
        # print("Exiting SendMessage state")
        pass



    def on_enter_WaitForResponse(self):
        # print("Entering WaitForResponse state")
        self.start_timeout_timer(WAIT_FOR_RESPONSE_TIMEOUT)
        self.wait_for_response()
        self.stop_timeout_timer()        
        if self.timeout:
            self.timeout = False            
            self.event_Timeout()
        else:       
            self.event_ResponseReceived()


        

    def on_exit_WaitForResponse(self):
        # print("Exiting WaitForResponse state")
        self.stop_timeout_timer()



    def on_enter_PlayResponse(self):
        # print("Entering PlayResponse state")
        self.play_message_on_speaker()
        self.event_Go()

    def on_exit_PlayResponse(self):
        # print("Exiting PlayResponse state")
        pass



    def on_enter_Timeout(self):
        # print("Entering Timeout state")
        self.timeout = True
        self.play_timeout_message_on_speaker()
        self.event_Go()

    def on_exit_Timeout(self):
        # print("Exiting Timeout state")
        pass



    def on_enter_CommError(self):
        # print("Entering CommError state")
        self.play_comm_error_message_on_speaker()
        self.event_Go()

    def on_exit_CommError(self):
        # print("Exiting CommError state")
        pass


    #######################################################################
    #  timeout functions


    def start_timeout_timer(self, duration:float):
        # print("Start Timeout Timer")
        self.timeout_thread = threading.Timer(duration, self.timeout_occurred)
        self.timeout_thread.start()  # after <duration> seconds, timeout_occurred will be called
        self.timeout = False

    def stop_timeout_timer(self):
        # print("Cancel Timeout Timer")        
        self.timeout_thread.cancel()

    def timeout_occurred(self):
        print("TIMEOUT occurred")
        self.timeout = True
        self.stop_timeout_timer()



    #######################################################################
    #  member functions


    def listen_for_hotwords(self) -> Optional[str]:
        self.hot_word_detected = False
        print("listen_for_hotwords")
        self.mic_stream.start_stream()

        while True :
            frame = self.mic_stream.getFrame()
            result = self.levi_hw.scoreFrame(frame)
            if result==None :
                #no voice activity
                continue
            if(result["match"]):
                print("Wakeword uttered",result["confidence"])
                # play the response
                self.audioplayer = AudioPlayer('/home/bryan/workspace/levi/audio/audioFiles/what_you_want.wav')
                self.audioplayer.play()
                self.audioplayer.close()                
                print("hotword heard")      
                return 'Levi'
               

    def start_recording(self):
        print("Start Recording Thread")        
        self.micrecorder.start_recording()

                

    def stop_recording(self):
        print("Stop Recording")
        self.micrecorder.stop_recording()
        

    
        

    def listen_for_voice(self):
        print("Listening for voice")
        
        chunk = 512  # Record in chunks of 1024 samples      
        fs = 16000  # Record at 44100 samples per second
                

        while not self.timeout:
            data = self.micrecorder.get_last_audio_chunk(chunk)
            
            data = np.array( [float(i)/32767.0 for i in data], dtype=np.float32)
            data1 = np.copy(data)

            if len(data1) > 0:

                # print(f'max = {data1.max()}    min = {data1.min()}     type = {type(data1[0])}')
                         
                t = torch.from_numpy(data1)

                # calculate probability that this chunk of audio containers speech
                speech_prob = self.silero_model(t, fs).item()  
                
                if speech_prob>0.5:
                    break
          
        # d = self.micrecorder.get_audio_data()
        # write_wav('question.wav',d,16000)
        # print("wav file written")

    def listen_for_silence(self):
        print("Listening for silence")        
        
        chunk = 512  # Record in chunks of 1024 samples     
        fs = 16000  # Record at 44100 samples per second
       
        # Store data in chunks for 3 seconds
        total_silence = 0.0
        while not self.timeout:
            data = self.micrecorder.get_last_audio_chunk(chunk)
            data = np.array( [float(i)/32767.0 for i in data], dtype=np.float32)
            data1 = np.copy(data)

            if len(data1) > 0:
                         
                t = torch.from_numpy(data1)
                time.sleep(0.032)
                # calculate probability that this chunk of audio containers speech
                speech_prob = self.silero_model(t, fs).item()  
            
                if speech_prob>0.5:
                    total_silence = 0.0
                else:
                    total_silence+=1.0
                if total_silence > 70.0:
                    break

        print(f"silence detected   {total_silence}")
        


    def send_message(self) -> bool:
        # convert to int16 data
        data = self.micrecorder.get_audio_data()
       
        write_wav("question.wav", data, 16000)

        print("Message Sent")
        time.sleep(1)
        return True

    def wait_for_response(self):
        print("Waiting for response")        
        response_received = False

        tot = 0.0
        while not response_received and not self.timeout and tot < 2:
            # check for response received, and set response_received appropriately            
            time.sleep(WAIT_UPDATE_DELAY)  
            tot += WAIT_UPDATE_DELAY
        

    def play_message_on_speaker(self):
        print("Play returned message on speaker")
        time.sleep(2)

    def play_timeout_message_on_speaker(self):
        print("Play timeout message on speaker")
        self.audioplayer = AudioPlayer('timeout.wav')
        self.audioplayer.play()
        self.audioplayer.close()   
        time.sleep(2)


    def play_comm_error_message_on_speaker(self):
        print("Comm Error")
        self.audioplayer = AudioPlayer('comm_error.wav')
        self.audioplayer.play()
        self.audioplayer.close()
        time.sleep(2)

###############################################################################
  


print("State Machine Started")
sm = AssistantStateMachine() # state machine started just by creating an instance

# sm._graph().write_png("/home/dssadmin/Pictures/state_machine.png")

# try:
#     while True:
#         time.sleep(2)
#         print("Still Looping...")
# except KeyboardInterrupt:
#     print('\ninterrupted!')

