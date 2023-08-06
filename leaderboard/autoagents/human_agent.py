#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""

import numpy as np
import json

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

from threading import Timer
import pandas as pd
import re
import carla
import openai
openai.api_key = 'PLACE_API_KEY_HERE'

import pinecone
index_name = 'driving-index'

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key="PLACE_API_KEY_HERE",
    environment="northamerica-northeast1-gcp"  # find next to api key in console
)
# check if 'openai' index already exists (only create index if not)
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=len(embeds[0]))
# connect to index
index = pinecone.Index(index_name)

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track


def get_entry_point():
    return 'HumanAgent'

class HumanInterface(object):

    """
    Class to control a vehicle manually for debugging purposes
    """

    def __init__(self, ego_vehicle, world, assistant, objects_path, audio_path, ctrl, width, height, side_scale, left_mirror=False, right_mirror=False):
        self._control = ctrl
        self._width = width
        self._height = height
        self._scale = side_scale
        self._surface = None

        self._left_mirror = left_mirror
        self._right_mirror = right_mirror

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Human Agent")

        self._font = pygame.freetype.SysFont(None, 28)
        self._font.origin=True

        self._fontsm = pygame.freetype.SysFont(None, 22)
        self._fontsm.origin=True

        self._time = 0
        self._speed = None
        self._ego_vehicle = ego_vehicle
        self._world = world
        self._objects_path = objects_path
        self._audio_path = audio_path
        self._collecting_data = True

        self._subtitle = None
        self._df = pd.read_excel(self._audio_path+"audio_output.xlsx")

        pygame.mixer.init()
        
        if (assistant):
            self._start_collect_data()

    def run_interface(self, input_data):
        """
        Run the GUI
        """

        # Process sensor data
        image_center = input_data['Center'][1][:, :, -2::-1]
        self._surface = pygame.surfarray.make_surface(image_center.swapaxes(0, 1))

        # Add the left mirror
        if self._left_mirror:
            image_left = input_data['Left'][1][:, :, -2::-1]
            left_surface = pygame.surfarray.make_surface(image_left.swapaxes(0, 1))
            self._surface.blit(left_surface, (0, (1 - self._scale) * self._height))

        # Add the right mirror
        if self._right_mirror:
            image_right = input_data['Right'][1][:, :, -2::-1]
            right_surface = pygame.surfarray.make_surface(image_right.swapaxes(0, 1))
            self._surface.blit(right_surface, ((1 - self._scale) * self._width, (1 - self._scale) * self._height))

        # Display image
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        
        self._speed = np.floor(3.6 * np.abs(input_data['speed'][1]['speed']))
        self._speed = str(int(self._speed))
        self._set_ctrl_interface()

        pygame.display.flip()

    def set_black_screen(self):
        """Set the surface to black"""
        black_array = np.zeros([self._width, self._height])
        self._surface = pygame.surfarray.make_surface(black_array)
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def _set_ctrl_interface(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT: return
        time_limit = (7 * 60 * 1000)
        ticks=time_limit - pygame.time.get_ticks() if time_limit > pygame.time.get_ticks() else pygame.time.get_ticks() - time_limit
        millis=ticks%1000
        seconds=int(ticks/1000 % 60)
        minutes=int(ticks/60000 % 24)
        out='{}{minutes:02d}:{seconds:02d}:{millis}'.format('' if time_limit > pygame.time.get_ticks() else '-', minutes=minutes, millis=millis, seconds=seconds)

        self._font.render_to(self._display, (100, 100), out, pygame.Color('white' if time_limit > pygame.time.get_ticks() else 'red'))

        # Display speed information
        self._fontsm.render_to(self._display, (100, 140), 'Speed: ', pygame.Color('white'))
        self._fontsm.render_to(self._display, (240, 140), self._speed, pygame.Color('white'))

        # Display gear information; 1 is normal and -1 is reverse
        out_gear='%s' % {-1: 'R', 0: 'N', 1: 'N'}.get(self._control.gear, self._control.gear)
        self._fontsm.render_to(self._display, (100, 170), 'Gear: ', pygame.Color('white'))
        self._fontsm.render_to(self._display, (240, 170), out_gear, pygame.Color('white'))

        # Display steering information
        steer=self._control.steer
        self._fontsm.render_to(self._display, (100, 200), 'Steering: ', pygame.Color('white'))

        rect_width = 80
        rect_height = 12
        fill_width = int(rect_width / 2 * np.abs(steer) / 0.7)

        pygame.draw.rect(self._display, pygame.Color('white'), (240, 190, rect_width, rect_height), 1)
        if (steer == 0):
            pygame.draw.rect(self._display, pygame.Color('white'), (240 + (rect_width // 2) - 1, 190, 2, rect_height))
        elif (steer > 0):
            pygame.draw.rect(self._display, pygame.Color('white'), (240 + rect_width // 2, 190, fill_width, rect_height))
        else:
            pygame.draw.rect(self._display, pygame.Color('white'), (240 + rect_width // 2 - fill_width, 190, fill_width, rect_height))

        # Display subtitle
        self._font.render_to(self._display, (100, self._height-60), self._subtitle if self._subtitle is not None else '', pygame.Color('white'))


    def _start_collect_data(self, prev_msg_time=0, prev_vehicle=False, prev_traffic=False, prev_walker=False):
        self._time = self._time+1
        msg_time = prev_msg_time
        vehicle = False
        traffic = False
        walker = False
        if (self._time > 20):
            s = self._collect_nearby_objects_and_control()

            if (type(s) == str):
                vehicle = re.search('vehicle', s) is not None
                traffic = re.search('traffic', s) is not None
                walker = re.search('walker', s) is not None
                is_new_objects = vehicle != prev_vehicle or traffic != prev_traffic or walker != prev_walker
                if self._subtitle is not None and self._time - prev_msg_time > 5:
                    self._subtitle = None

                if (is_new_objects and self._time - prev_msg_time >= 20) or self._time - prev_msg_time >= 30:
                    try:
                        emb = self._get_embedding(s)
                        msg = self._query_with_embedding(emb)

                        path = msg['matches'][0]['id']
                        msg_time = self._time

                        print('Play audio at time {}: {}'.format(msg_time, path))
                        sound_effect = pygame.mixer.Sound(self._audio_path+path)
                        sound_effect.play()

                        df_ref = self._df.loc[self._df['path'] == path]
                        self._subtitle = df_ref['message'].values[0]

                    except Exception as e:
                        print("An error occurred while processing the objects:", str(e))

        Timer(1, self._start_collect_data, args=(msg_time, vehicle, traffic, walker,)).start()

    def _collect_nearby_objects_and_control(self):
        threshold = 50
        t = self._ego_vehicle.get_transform()
        obj = self._world.get_actors()

        if len(obj) > 1:
            distance = lambda l: np.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            # Show nearby actors within a distance threshold
            
            obj = [(int(np.floor(distance(x.get_location()))), x.type_id) 
                        for x in obj if x.id != self._ego_vehicle.id and (distance(x.get_location()) <= threshold)]
            
        ctrl = self._control
        spd = self._speed
        spd_limit = self._ego_vehicle.get_speed_limit()
        s = self._write_file(obj, ctrl, spd, spd_limit)

        return s

    def _write_file(self, objects, ctrl, spd, spd_limit):
        if (len(objects) > 1):
            objects = [(dist, x) for (dist, x) in objects if x.startswith('static') == False and x.startswith('sensor') == False]
            ctrl = "['throttle':{}, 'steer':{}, 'brake':{}, 'speed':{}, 'speedLimit':{}]\n\n".format(
                        0.7 if ctrl.throttle > 0 else 0,
                        max(-1, min(1, ctrl.steer)),
                        ctrl.hand_brake or ctrl.brake > 0,
                        spd,
                        spd_limit
            )

            # TODO: replace here with args from leaderboard evaluator
            with open(self._objects_path, 'a') as f:
                f.write(
                    "count: {}\n"
                    "objects: {}\n"
                    "ctrl: {}".format(
                        len(objects), 
                        objects,
                        ctrl
                    )
                )

            return "{}{}".format(objects, ctrl.replace('\n',''))

    def _get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
    
    def _query_with_embedding(self, emb, k=1):
        results = index.query(
            vector=emb,
            top_k=k,
            include_values=False
        )
        return results

    def _quit(self):
        pygame.quit()


class HumanAgent(AutonomousAgent):

    """
    Human agent to control the ego vehicle via keyboard
    """

    current_control = None
    agent_engaged = False

    def setup(self, path_to_conf_file, ego_vehicle, world, assistant=False, objects_path="./objects_detected.txt", audio_path="./audio/"):
        """
        Setup the agent parameters
        """
        self._control = carla.VehicleControl()
        self.track = Track.SENSORS

        self.agent_engaged = False
        self.camera_width = 1280
        self.camera_height = 720
        self._side_scale = 0.3
        self._left_mirror = False
        self._right_mirror = False

        self._ego_vehicle = ego_vehicle
        self._world = world
        self._objects_path = objects_path
        self._audio_path = audio_path

        self._hic = HumanInterface(
            self._ego_vehicle,
            self._world,
            assistant,
            self._objects_path,
            self._audio_path,
            self._control,
            self.camera_width,
            self.camera_height,
            self._side_scale,
            self._left_mirror,
            self._right_mirror
        )
        self._controller = KeyboardControl(self._control, path_to_conf_file)
        self._prev_timestamp = 0

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]
        """

        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.9, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': self.camera_width, 'height': self.camera_height, 'fov': 120, 'id': 'Center'},
            {'type': 'sensor.speedometer','reading_frequency': 25, 'id': 'speed'}
        ]

        if self._left_mirror:
            sensors.append(
                {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -1.0, 'z': 1, 'roll': 0.0, 'pitch': 0.0, 'yaw': 210.0,
                 'width': self.camera_width * self._side_scale, 'height': self.camera_height * self._side_scale,
                 'fov': 100, 'id': 'Left'})

        if self._right_mirror:
            sensors.append(
                {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 1.0, 'z': 1, 'roll': 0.0, 'pitch': 0.0, 'yaw': 150.0,
                 'width': self.camera_width * self._side_scale, 'height': self.camera_height * self._side_scale,
                 'fov': 100, 'id': 'Right'})

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        self.agent_engaged = True
        self._hic.run_interface(input_data)

        control = self._controller.parse_events(timestamp - self._prev_timestamp)
        self._prev_timestamp = timestamp

        return control

    def destroy(self):
        """
        Cleanup
        """
        self._hic.set_black_screen()
        self._hic._quit = True


class KeyboardControl(object):

    """
    Keyboard control for the human agent
    """

    def __init__(self, ctrl, path_to_conf_file):
        """
        Init
        """
        self._control = ctrl
        self._steer_cache = 0.0
        self._clock = pygame.time.Clock()

        # Get the mode
        if path_to_conf_file:

            with (open(path_to_conf_file, "r")) as f:
                lines = f.read().split("\n")
                self._mode = lines[0].split(" ")[1]
                self._endpoint = lines[1].split(" ")[1]

            # Get the needed vars
            if self._mode == "log":
                self._log_data = {'records': []}

            elif self._mode == "playback":
                self._index = 0
                self._control_list = []

                with open(self._endpoint) as fd:
                    try:
                        self._records = json.load(fd)
                        self._json_to_control()
                    except json.JSONDecodeError:
                        pass
        else:
            self._mode = "normal"
            self._endpoint = None

    def _json_to_control(self):

        # transform strs into VehicleControl commands
        for entry in self._records['records']:
            control = carla.VehicleControl(throttle=entry['control']['throttle'],
                                           steer=entry['control']['steer'],
                                           brake=entry['control']['brake'],
                                           hand_brake=entry['control']['hand_brake'],
                                           reverse=entry['control']['reverse'],
                                           manual_gear_shift=entry['control']['manual_gear_shift'],
                                           gear=entry['control']['gear'])
            self._control_list.append(control)

    def parse_events(self, timestamp):
        """
        Parse the keyboard events and set the vehicle controls accordingly
        """
        # Move the vehicle
        if self._mode == "playback":
            self._parse_json_control()
        else:
            self._parse_vehicle_keys(pygame.key.get_pressed(), timestamp*1000)

        # Record the control
        if self._mode == "log":
            self._record_control()

        return self._control

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        Calculate new vehicle controls based on input keys
        """

        if keys[K_q]:
            self._control.gear = 1 if self._control.reverse else -1
            self._control.reverse = self._control.gear < 0

        if keys[K_UP] or keys[K_w]:
            self._control.throttle = 0.7
        else:
            self._control.throttle = 0.0
        
        steer_increment = 3e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                if self._steer_cache == 0:
                    self._steer_cache = -0.05
                else:
                    self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                if self._steer_cache == 0:
                    self._steer_cache = 0.05
                else:
                    self._steer_cache += steer_increment
        else:
            if self._steer_cache > 0:
                self._steer_cache = max(0, self._steer_cache - steer_increment * 1.4)
            else:
                self._steer_cache = min(0, self._steer_cache + steer_increment * 1.4)
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))

        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_json_control(self):

        if self._index < len(self._control_list):
            self._control = self._control_list[self._index]
            self._index += 1
        else:
            print("JSON file has no more entries")

    def _record_control(self):
        new_record = {
            'control': {
                'throttle': self._control.throttle,
                'steer': self._control.steer,
                'brake': self._control.brake,
                'hand_brake': self._control.hand_brake,
                'reverse': self._control.reverse,
                'manual_gear_shift': self._control.manual_gear_shift,
                'gear': self._control.gear
            }
        }

        self._log_data['records'].append(new_record)

    def __del__(self):
        # Get ready to log user commands
        if self._mode == "log" and self._log_data:
            with open(self._endpoint, 'w') as fd:
                json.dump(self._log_data, fd, indent=4, sort_keys=True)
