import sys
import pygame

from XuanJing.gamecore.cards.guandan.pygame_agentbase import GuanDanPyGame, Card


class PyGameAgent(GuanDanPyGame):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self):
        super(PyGameAgent, self).__init__()

    # @staticmethod
    def step(self, state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''
        # # cards = 'D2 H3 S4 D4 C4 C4 D4 H4 C5 D5 C6 S6 H7 H7 H8 S8 C9 H9 ST CJ DJ SJ HJ DK SA SA CA'
        cards = state.player_hand_cards
        self.cardsInHand = [Card(card[0], card[1]) for card in cards.split(" ")]
        while True:

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # 鼠标点击触发
                    self.clock.tick(self.FPS)
                    self.select(event)  # 鼠标选牌

                    if self.playButton.collidepoint(event.pos):  # 检测到点击‘出牌按钮’
                        # TODO hzq 选择的牌位未花色
                        have_selected = "".join([selectcard.rank for selectcard in self.selectedCards])
                        if have_selected in state.legal_action:
                            self.selectedCards = []
                            return have_selected
                        else:
                            for card in self.selectedCards:
                                card.isSelected = False
                            self.selectedCards = []
                    if self.passButton.collidepoint(event.pos):  # 检测到点击‘pass’按钮
                        self.selectedCards = []
                        return 'pass'

                if event.type == pygame.QUIT:  # 检测到关闭游戏界面按钮
                    pygame.quit()
                    sys.exit()
            print([selectcard.rank for selectcard in self.selectedCards])
            screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            mousePos = pygame.mouse.get_pos()
            self.draw_basics(screen, mousePos, None)
            self.draw_cards(screen)

            # pygame 循环渲染各个玩家的动作信息
            for player_id, played_card in state.history.items():
                if len(played_card) != 0:
                    if played_card[-1] != "pass":
                        self.draw_cards_played(screen, player_id, state.player_id, played_card[-1])  # 出掉的牌
                    else:
                        self.draw_pass(screen, player_id, state.player_id, played_card[-1])  # 绘制Pass文字
                self.dra_num_cards_left(screen, player_id, state.player_id, state.num_cards_left[player_id])

            pygame.display.set_caption("启元世界-掼蛋!")
            pygame.display.update()

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        # probs = [0 for _ in range(self.num_actions)]
        # for i in state['legal_actions']:
        #     probs[i] = 1/len(state['legal_actions'])

        info = {}
        # info['probs'] = {state['raw_legal_actions'][i]: probs[state['legal_actions'][i]] for i in range(len(state['legal_actions']))}

        return self.step(state), info