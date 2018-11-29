# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

from atari_zoo.translate import translate_game_name

def get_random_agent_scores(game):
    global random_play_scores

    translated_name = translate_game_name(game,'canonical','apex')

    return random_play_scores[translated_name]

def get_human_scores(game):
    global human_scores

    translated_name = translate_game_name(game,'canonical','apex')

    return human_scores[translated_name]

human_scores = {
    "alien": 6875,
    "amidar": 1676,
    "assault": 1496,
    "asterix": 8503,
    "asteroids": 13157,
    "atlantis": 29028,
    "bank_heist": 734.4,
    "battle_zone": 37800,
    "beam_rider": 5775,
    "bowling": 154.8,
    "boxing": 4.3,
    "breakout": 31.8,
    "centipede": 11963,
    "chopper_command": 9882,
    "crazy_climber": 35411,
    "demon_attack": 3401,
    "double_dunk": -15.5,
    "enduro": 309.6,
    "fishing_derby": 5.5,
    "freeway": 29.6,
    "frostbite": 4335,
    "gopher": 2321,
    "gravitar": 2672,
    "hero": 25763,
    "ice_hockey": 0.9,
    "jamesbond": 406.7,
    "kangaroo": 3035,
    "krull": 2395,
    "kung_fu_master": 22736,
    "montezuma_revenge": 4367,
    "ms_pacman": 15693,
    "name_this_game": 4076,
    "pong": 9.3,
    "private_eye": 69571,
    "qbert": 13455,
    "riverraid": 13513,
    "road_runner": 7845,
    "robotank": 11.9,
    "seaquest": 20182,
    "space_invaders": 1652,
    "star_gunner": 10250,
    "tennis": -8.9,
    "time_pilot": 5925,
    "tutankham": 167.6,
    "up_n_down": 9082,
    "venture": 1188,
    "video_pinball": 17298,
    "wizard_of_wor": 4757,
    "zaxxon": 9173,
}

random_play_scores = {
    "alien": 227.8,
    "amidar": 5.8,
    "assault": 222.4,
    "asterix": 210,
    "asteroids": 719.1,
    "atlantis": 12850,
    "bank_heist": 14.2,
    "battle_zone": 2360,
    "beam_rider": 363.9,
    "bowling": 23.1,
    "boxing": 0.1,
    "breakout": 1.7,
    "centipede": 2091,
    "chopper_command": 811,
    "crazy_climber": 10781,
    "demon_attack": 152.1,
    "double_dunk": -18.6,
    "enduro": 0,
    "fishing_derby": -91.7,
    "freeway": 0,
    "frostbite": 65.2,
    "gopher": 257.6,
    "gravitar": 173,
    "hero": 1027,
    "ice_hockey": -11.2,
    "jamesbond": 29,
    "kangaroo": 52,
    "krull": 1598,
    "kung_fu_master": 258.5,
    "montezuma_revenge": 0,
    "ms_pacman": 307.3,
    "name_this_game": 2292,
    "pong": -20.7,
    "private_eye": 24.9,
    "qbert": 163.9,
    "riverraid": 1339,
    "road_runner": 11.5,
    "robotank": 2.2,
    "seaquest": 68.4,
    "space_invaders": 148,
    "skiing":-16679.9,
    "star_gunner": 664,
    "tennis": -23.8,
    "time_pilot": 3568,
    "tutankham": 11.4,
    "up_n_down": 533.4,
    "venture": 0,
    "video_pinball": 16257,
    "wizard_of_wor": 563.5,
    "zaxxon": 32.5,
}

