import random
from numpy import array, argmax
from config_reader import games, attemps
from statistics import median


def train_population(env, model=None):
      training_data = []
      all_scores = []
      accepted_scores = []
      game_memory = []
      number_of_observations = 0
      print("playing games...")
      for game_num in range(games):
            # game starts here
            env.reset()
            score = 0
            round_memory = []
            for step, attempt_num in enumerate(range(attemps)):
                  # just one more step
                  if model is None or step == 0:
                        action = random.randrange(0, 2)
                  else:
                        """
                        print(array(observation).reshape(-1, number_of_observations))
                        print(model.predict(array(observation).reshape(-1, number_of_observations)))
                        print(argmax(model.predict(array(observation).reshape(-1, number_of_observations))))
                        """
                        action = argmax(model.predict(
                              array(observation).reshape(-1, number_of_observations)))
                  observation, reward, done, info = env.step(action)
                  number_of_observations = len(observation)
                  score += reward
                  data = (observation, action)
                  round_memory.append(data)

                  if done:
                        print(score)
                        break
            round_memory.insert(0, score)

            game_memory.append(round_memory)

      top_games = int(games * .05)

      [all_scores.append(score[0]) for score in game_memory]
      min_requires_score = sorted(all_scores, reverse=True)[top_games]

      # creating table for actions
      output = [0] * env.action_space.n

      for game in game_memory:
            score = game[0]
            if score >= min_requires_score:
                  accepted_scores.append(score)
                  for action_observation in game[1:]:
                        action_num = action_observation[1]
                        output[action_num] = 1
                        training_data.append([action_observation[0], output])
                        # reset table of actions
                        output = [0] * env.action_space.n

      median_accepted_score = median(accepted_scores)
      median_score = median(all_scores)

      return median_score, median_accepted_score, training_data, number_of_observations, env
