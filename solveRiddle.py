from prisonersEnv import Prison
import dqn

if __name__ == "__main__":

	nPrisoners = 10
	hidden_dims = 128
	nActions = 3
	features = nPrisoners + nActions + 1
	learner = dqn.learner(features, hidden_dims, nActions)

	folsom = Prison(learner, nPrisoners)

	num_episodes = 10000
	for episode in range(num_episodes):
		folsom.visit()

