import sys
import pygame


class Card(object):
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
        self.image = pygame.image.load(f"C:/guandan/Guandan/CFR/pukeImage/{suit + rank}.png")
        self.isSelected = False

    def __str__(self):
        if self.rank == 10:
            rank_str = "T"
        elif self.rank == 11:
            rank_str = "J"
        elif self.rank == 12:
            rank_str = "Q"
        elif self.rank == 13:
            rank_str = "K"
        elif self.rank == 14:
            rank_str = "A"
        elif self.rank == 15:
            rank_str = "Small Joker"
        elif self.rank == 16:
            rank_str = "Big Joker"
        else:
            rank_str = str(self.rank)
        return f"{self.suit}{rank_str}"


class GuanDanPyGame(object):
    def __init__(self):
        pygame.init()
        self.FPS = 30
        self.clock = pygame.time.Clock()

        self.WIDTH = 1024
        self.HEIGHT = 768
        self.green = (0, 200, 0)
        self.red = (200, 0, 0)
        self.yellow = (240, 220, 0)
        self.brightYellow = (255, 255, 0)
        self.brightGreen = (0, 255, 0)
        self.brightRed = (255, 0, 0)

        self.bg = "C:/guandan/Guandan/CFR/bg1.png"

        self.cardsInHand = []
        self.selectedCards = []

    def run(self):
        cards = 'D2 H3 S4 D4 C4 C4 D4 H4 C5 D5 C6 S6 H7 H7 H8 S8 C9 H9 ST CJ DJ SJ HJ DK SA SA CA'
        self.cardsInHand = [Card(card[0], card[1]) for card in cards.split(" ")]
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.select(event)
                    # TODO 判断是否为合法动作
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            mousePos = pygame.mouse.get_pos()
            self.drawBasics(screen, mousePos, None)
            self.drawCards(screen)
            pygame.display.set_caption("启元世界-掼蛋")
            pygame.display.update()
            # pygame.display.flip()  # 更新屏幕内容

    def dra_num_cards_left(self, screen, relative_id, player_id, num_left):
        font = pygame.font.Font(None, 30)
        playerText = font.render(f"PLAYER {relative_id}", True, (255,) * 3)
        text = font.render(f"{num_left} cards left", True, (255,) * 3)
        index = (relative_id + 4 - player_id) % 4
        if index == 0:
            startX, startY = self.WIDTH / 2 - 55, 720
        elif index == 1:
            startX, startY = 895, self.HEIGHT / 2 - 100
        elif index == 2:
            startX, startY = self.WIDTH / 2 - 55, 20
        else:
            startX, startY = 15, self.HEIGHT / 2 - 100
        screen.blit(playerText, (startX, startY))
        screen.blit(text, (startX, startY + 20))

    def draw_pass(self, screen, relative_id, player_id, cardsPlayed):
        font = pygame.font.Font(None, 30)
        index = (relative_id + 4 - player_id) % 4
        if index == 0:
            startX, startY = self.WIDTH / 2 - ((len(cardsPlayed) - 1) * 20 + 105) / 2, 450
        elif index == 1:
            startX, startY = 750, self.HEIGHT / 2 - 130
        elif index == 2:
            startX, startY = self.WIDTH / 2 - ((len(cardsPlayed) - 1) * 20 + 105) / 2, 130
        else:
            startX, startY = 150, self.HEIGHT / 2 - 130
        passText = font.render('PASS', True, (191,) * 3)
        screen.blit(passText, (startX, startY))

    def draw_cards_played(self, screen, relative_id, player_id, cardsPlayed):
        index = (relative_id + 4 - player_id) % 4
        if index == 0:
            startX, startY = self.WIDTH / 2 - ((len(cardsPlayed) - 1) * 20 + 105) / 2, 330
        elif index == 1:
            startX, startY = 690, self.HEIGHT / 2 - 130
        elif index == 2:
            startX, startY = self.WIDTH / 2 - ((len(cardsPlayed) - 1) * 20 + 105) / 2, 130
        else:
            startX, startY = 150, self.HEIGHT / 2 - 130
        for i in range(len(cardsPlayed)):
            # TODO hzq 修正卡牌花色
            card = cardsPlayed[i]
            if card == "J":
                card = Card(suit="B", rank=card)
            else:
                card = Card(suit="D", rank=card)
            screen.blit(card.image, (startX + 20 * i, startY))

    def select(self, event):
        startX = self.WIDTH / 2 - ((len(self.cardsInHand) - 1) * 20 + 105) / 2
        startY = 560
        x, y = event.pos
        if startX < x < startX + 20 * (len(self.cardsInHand) - 1) + 105:
            i = -1 if x > startX + 20 * (len(self.cardsInHand) - 1) else int((x - startX) // 20)
            if self.cardsInHand[i].isSelected:
                if startY - 20 < y < startY + 130:
                    self.cardsInHand[i].isSelected = False
            else:
                if startY < y < startY + 160:
                    self.cardsInHand[i].isSelected = True
        self.selectedCards = [card for card in self.cardsInHand if card.isSelected]

    def draw_cards(self, screen):
        startX = self.WIDTH / 2 - ((len(self.cardsInHand) - 1) * 20 + 105) / 2
        startY = 560
        for i in range(len(self.cardsInHand)):
            card = self.cardsInHand[i]
            if card.isSelected:
                screen.blit(card.image, (startX + 20 * i, startY - 20))
            else:
                screen.blit(card.image, (startX + 20 * i, startY))

    def draw_basics(self, screen, mousePos, state):
        font = pygame.font.Font(None, 24)
        self.passButton = pygame.Rect(self.WIDTH / 2 - 100, 500, 60, 30)
        self.playButton = pygame.Rect(self.WIDTH / 2 - 190, 500, 60, 30)
        self.hintButton = pygame.Rect(self.WIDTH / 2 - 10, 500, 60, 30)
        self.deseletButton = pygame.Rect(self.WIDTH / 2 + 80, 500, 100, 30)
        playText = font.render("Play", True, (0,) * 3)
        passText = font.render("Pass", True, (0,) * 3)
        hintText = font.render("Hint", True, (0,) * 3)
        deselectText = font.render("Deselect All", True, (0,) * 3)
        self.draw_background(screen)
        if self.passButton.collidepoint(mousePos):
            pygame.draw.rect(screen, self.brightGreen, self.passButton)
        else:
            pygame.draw.rect(screen, self.green, self.passButton)
        if self.playButton.collidepoint(mousePos):
            pygame.draw.rect(screen, self.brightRed, self.playButton)
        else:
            pygame.draw.rect(screen, self.red, self.playButton)
        if self.hintButton.collidepoint(mousePos):
            pygame.draw.rect(screen, self.brightYellow, self.hintButton)
        else:
            pygame.draw.rect(screen, self.yellow, self.hintButton)
        if self.deseletButton.collidepoint(mousePos):
            pygame.draw.rect(screen, self.brightRed, self.deseletButton)
        else:
            pygame.draw.rect(screen, self.red, self.deseletButton)

        screen.blit(passText, (self.WIDTH / 2 - 89, 507))
        screen.blit(playText, (self.WIDTH / 2 - 178, 507))
        screen.blit(hintText, (self.WIDTH / 2 + 2, 507))
        screen.blit(deselectText, (self.WIDTH / 2 + 83, 507))

    def draw_background(self, screen):
        bg = pygame.image.load(self.bg)
        screen.blit(bg, (0, 0))


if __name__ == "__main__":
    GuanDanPyGame().run()