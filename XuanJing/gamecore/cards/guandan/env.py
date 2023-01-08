from XuanJing.gamecore.cards.guandan.game.game import GuandanGame as Game


class Env(object):
    """
    The base Env class for RL,
    we should base on this class and implement as many functions
    as we can.
    """
    def __init__(self, agents):
        """
        Initialize the environment.
        Args:
            agents (list): List of Agent classes, Set the agents
            that will interact with the environment.
        """

        self.game = Game()
        self.agents = agents

    def reset(self):
        """
        Start a new Game.
        """
        state = self.game.reset()
        return state

    def step(self, action):
        """
        Step forward:
        Args:
            action: The action taken by the current player
        Returns:
            next state of env.
        """

        next_state = self.game.step(action)

        return next_state

    def run(self):
        """
        Run a complete game.
        """

        state = self.reset()

        while not self.game.is_over():
            player_id = state.player_id
            action, _ = self.agents[player_id].eval_step(state)
            next_state = self.step(action)
            state = next_state

        # Payoffs
        payoffs = self.get_payoffs()

        return [], payoffs

    def get_payoffs(self):
        """
        Get the payoffs of players. Must be implemented in the child class.
        Returns:
            (list): A list of payoffs for each player.

        """
        return self.game.judger.judge_payoffs(self.game.winner_id)
