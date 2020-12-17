from train_population import train_population
from nn import neural_network
from gym import make
from config_reader import env_name, generations
from pandas import DataFrame


env = make(env_name)
median_scores_list = []
render = False


if __name__ == '__main__':
      median_scores, training_data, number_of_observations, env = train_population(
            env=env)
      model = neural_network(
            training_data=training_data,
            number_of_observations=number_of_observations)
      for generation in range(generations):
            if generation == generations:
                  render = True

            median_scores, training_data, number_of_observations, env = train_population(
                  env=env,
                  model=model,
                  number_of_observations=number_of_observations,
                  render=render)
            model = neural_network(
                  training_data=training_data,
                  number_of_observations=number_of_observations)

            median_scores_list.append(median_scores)
            print("generation: {}/{}".format(generation, median_scores))

            df = DataFrame({
                  "score": median_scores_list
            })
      print(df)
