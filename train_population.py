import random
from numpy import array, save, argmax
from config_reader import games, attemps, elitism_factor
from statistics import median


def train_population(env, render=False, model=None, number_of_observations=None):
      training_data = []
      all_games = []
      all_scores = []
      accepted_scores = []
      for game_num in range(games):
            env.reset()
            score = 0
            game_memory = []
            previous_memory = []
            for step, attempt_num in enumerate(range(attemps)):
                  if render:
                        env.render()
                  if model is None:
                        action = random.randrange(0, 2)
                  else:
                        if step == 0:
                              action = random.randrange(0, 2)
                        elif step > 0:
                              action = argmax(model.predict(
                                    array(observation).reshape(-1, number_of_observations)))
                  observation, reward, done, info = env.step(action)
                  number_of_observations = len(env.step(action)[0])
                  score += reward
                  data_to_extend = [[list(observation), action]]
                  if game_memory == []:
                        game_memory = data_to_extend
                  else:
                        game_memory.extend(data_to_extend)

                  previous_memory = game_memory

                  if done:
                        break

                  if render:
                        env.reset()


            print("game: {}/{}".format(game_num, score))
            all_games.extend([[previous_memory, score]])

      [all_scores.append(score[1]) for score in all_games]

      sorted_scores = sorted(all_scores, reverse=True)
      elitism_num = int(len(all_scores) * elitism_factor)
      score_requirements = sorted_scores[elitism_num]

      for game in all_games:
            score = game[1]
            if score >= score_requirements:
                  accepted_scores.append(score)
                  output = "no_action"
                  for action_observation in game[0]:
                        if action_observation[1] == 0:
                              output = [0, 1]
                        elif action_observation[1] == 1:
                              output = [1, 0]
                        training_data.append([action_observation[0], output])

      training_data_array = array(training_data)
      save("training", training_data_array)

      return median(accepted_scores), training_data, number_of_observations, env
