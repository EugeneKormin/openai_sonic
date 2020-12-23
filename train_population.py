import random
from numpy import array, argmax
from config_reader import games, attemps, threshold_coeff
from statistics import median, mean
import statistics


def train_population(env, model=None):
      training_data = []
      all_games = []
      all_scores = []
      accepted_scores = []
      all_steps_list = []
      number_of_observations = 0
      print("playing games...")
      for game_num in range(games):
            env.reset()
            score = 0
            game_memory = []
            previous_memory = []
            for step, attempt_num in enumerate(range(attemps)):
                  if model is None or step == 0:
                        action = random.randrange(0, 2)
                  else:
                        action = argmax(model.predict(
                              array(observation).reshape(-1, number_of_observations)))
                  observation, reward, done, info = env.step(action)
                  number_of_observations = len(observation)
                  score += reward
                  data_to_extend = [[list(observation), action]]
                  if game_memory == []:
                        game_memory = data_to_extend
                  else:
                        game_memory.extend(data_to_extend)

                  previous_memory = game_memory

                  if done:
                        break

            all_steps_list.append(len(previous_memory))
            all_games.extend([[previous_memory, score]])

      '''      
      median_attemps_in_game = median(data=all_steps_list)
      attemps_to_take_into_consideration = median_attemps_in_game * top_games
      '''

      top_games = int(games * .05)
      [all_scores.append(score[1]) for score in all_games]
      min_requires_score = sorted(all_scores, reverse=True)[top_games]

      # creating table for actions
      output = [0] * env.action_space.n

      for game in all_games:
            score = game[1]
            if score >= min_requires_score:
                  accepted_scores.append(score)
                  for action_observation in game[0]:
                        action_num = action_observation[1]
                        output[action_num] = 1
                        training_data.append([action_observation[0], output])
                        # reset table of actions
                        output = [0] * env.action_space.n

      median_accepted_score = median(accepted_scores)
      median_score = median(all_scores)

      return median_score, median_accepted_score, training_data, number_of_observations, env
