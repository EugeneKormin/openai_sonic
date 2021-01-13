from train_population import train_population
from gym import make
from config_reader import env_name, generations
from pandas import DataFrame
from nn import neural_network


env = make(env_name)
median_scores_list = []
median_best_score_list = []
render = False


if __name__ == '__main__':
      # train initial (random) population
      median_score, median_accepted_scores, training_data, number_of_observations, env = train_population(env=env)
      model = neural_network(
            training_data=training_data,
            number_of_observations=number_of_observations,
            number_of_actions=env.action_space.n
      )
      median_best_score_list.append(median_accepted_scores)
      median_scores_list.append(median_score)
      print("generation: {} / score: {}".format("0", median_accepted_scores))

      df = DataFrame({
            "best scores": median_best_score_list,
            "median score": median_scores_list
      })
      print(df)
      for generation in range(generations):
            # improving existing population
            median_score, median_accepted_scores, training_data, number_of_observations, env = train_population(
                  env=env,
                  model=model
            )

            model = neural_network(
                  training_data=training_data,
                  number_of_observations=number_of_observations,
                  number_of_actions=env.action_space.n
            )

            median_best_score_list.append(median_accepted_scores)
            median_scores_list.append(median_score)
            print("generation: {}/score: {}".format(generation + 1, median_accepted_scores))

            df = DataFrame({
                  "best scores": median_best_score_list,
                  "median score": median_scores_list
            })
            print(df)
