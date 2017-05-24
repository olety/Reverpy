import random
import sys
import pygame
import math
import time
import copy
import numpy as np
from pygame.locals import MOUSEBUTTONUP, QUIT
from enum import IntEnum
import json


class PlayerType(IntEnum):
    COMPUTER = 0
    PERSON = 1


class AiType(IntEnum):
    PERSON = -1
    NAIVE = 0
    MINMAX = 1
    ABETA = 2


class TileType(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    @property
    def inverse(self):
        if self.value == 0:  # Can't invert an empty cell
            return self
        elif self.value == 1:  # Black tile, can be achieved with .name == 'BLACK'
            return TileType.WHITE
        elif self.value == 2:  # White tile
            return TileType.BLACK

    @property
    def color(self):
        if self.value == 0:
            return None
        elif self.value == 1:  # Black tile, can be achieved with .name == 'BLACK'
            return (0, 0, 0)
        elif self.value == 2:  # White tile
            return (255, 255, 255)

    @property
    def placed(self):
        return self.value == 1 or self.value == 2


# class Tree:
#     def __init__(self, )


class Player:
    @property
    def computer(self):
        return self.type == PlayerType.COMPUTER

    @property
    def person(self):
        return self.type == PlayerType.PERSON

    def __init__(self, type, opponent, tiletype=TileType.EMPTY, aitype=AiType.PERSON):
        self.type = type
        self.aitype = aitype
        self.tiletype = tiletype
        self.opponent = opponent

    def update_tiletype(self, tiletype):
        self.tiletype = tiletype

    def update_aitype(self, aitype):
        self.aitype = aitype

    def move(self, board):
        if self.person:
            return False
        if self.aitype == AiType.NAIVE:
            return self.naive_move(board)
        elif self.aitype == AiType.MINMAX:
            return self.minmax_move(board, max_depth=3)
        elif self.aitype == AiType.ABETA:
            return self.abeta_move(board, max_depth=3)

    def minmax_move(self, board, max_depth):
        score, x, y = self.minmax(board, tiletype=self.tiletype,
                                  curr_depth=1, max_depth=max_depth)
        return Move(self, x, y)

    def minmax(self, board, tiletype, curr_depth, max_depth):
        if curr_depth >= max_depth:
            return board.get_score_heuristic_(tiletype),
        legal_moves = board.get_legal_moves_pos_(tiletype)
        if not legal_moves:
            return board.get_score_heuristic_(tiletype),
        if tiletype == self.tiletype:  # Maximizing
            best_score = -1000000
            for x, y in legal_moves:
                if board.calc_move_(x, y, tiletype):
                    score = self.minmax(board, tiletype.inverse,
                                        curr_depth + 1, max_depth)[0]
                    board.undo_last()
                    if score > best_score:
                        best_score, best_x, best_y = score, x, y
                else:
                    return board.get_score_heuristic_(tiletype),
        else:  # Minimizing
            best_score = 1000000
            for x, y in legal_moves:
                if board.calc_move_(x, y, tiletype):
                    score = self.minmax(board, tiletype.inverse,
                                        curr_depth + 1, max_depth)[0]
                    board.undo_last()
                    if score < best_score:
                        best_score, best_x, best_y = score, x, y
                else:
                    return board.get_score_heuristic_(tiletype),
        return best_score, best_x, best_y

    def abeta_move(self, board, max_depth):
        score, x, y = self.abeta(board, tiletype=self.tiletype,
                                 curr_depth=1, max_depth=max_depth,
                                 a=-1000000, b=1000000)
        return Move(self, x, y)

    def abeta(self, board, tiletype, curr_depth, max_depth, a, b):
        if curr_depth >= max_depth:
            return board.get_score_heuristic_(tiletype),
        legal_moves = board.get_legal_moves_pos_(tiletype)
        if not legal_moves:
            return board.get_score_heuristic_(tiletype),
        if tiletype == self.tiletype:  # Maximizing
            best_score = -1000000
            for x, y in legal_moves:
                if board.calc_move_(x, y, tiletype):
                    score = self.abeta(board, tiletype.inverse,
                                       curr_depth + 1, max_depth, a, b)[0]
                    board.undo_last()
                    if score > best_score:
                        best_score, best_x, best_y = score, x, y
                    a = max(a, best_score)
                    if b <= a:
                        continue  # B cut-off
                else:
                    return board.get_score_heuristic_(tiletype),
        else:  # Minimizing
            best_score = 1000000
            for x, y in legal_moves:
                if board.calc_move_(x, y, tiletype):
                    score = self.abeta(board, tiletype.inverse,
                                       curr_depth + 1, max_depth, a, b)[0]
                    board.undo_last()
                    if score < best_score:
                        best_score, best_x, best_y = score, x, y
                    b = min(b, best_score)
                    if b <= a:
                        continue  # A cut-off
                else:
                    return board.get_score_heuristic_(tiletype),
        return best_score, best_x, best_y

    def naive_move(self, board):
        if self.person:
            return False

        # Returns a movelist
        legal_moves = board.get_legal_moves(self)

        # always go for a corner if available.
        for move in legal_moves:
            if board._check_corner(move.x, move.y):
                return move

        # randomize the order of the possible moves
        random.shuffle(legal_moves)

        # Go through all possible moves and remember the best scoring move
        best_score = -1
        for move in legal_moves:
            # TODO: change deepcopy to something better
            next_iter_board = copy.deepcopy(board)
            next_iter_board.calc_move(move)
            score = next_iter_board.get_score(self)
            if score > best_score:
                best_move, best_score = move, score
        return best_move


class Move:
    def __init__(self, player, x, y):
        self.player = player
        self.x = x
        self.y = y


class Board:
    # (X,Y)
    MOVE_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1),
                       (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    SCOREBOARD = (
        (7, 2, 5, 4, 4, 5, 2, 7),
        (2, 1, 3, 3, 3, 3, 1, 2),
        (5, 3, 6, 5, 5, 6, 3, 5),
        (4, 3, 5, 6, 6, 5, 3, 4),
        (4, 3, 5, 6, 6, 5, 3, 4),
        (5, 3, 6, 5, 5, 6, 3, 3),
        (2, 1, 3, 3, 3, 3, 1, 2),
        (7, 2, 5, 4, 4, 5, 2, 7),
    )  # , dtype=np.dtype('int8'))

    @property
    def cols(self):
        return self.board.shape[1]

    @property
    def rows(self):
        return self.board.shape[0]

    def at(self, x, y):
        return TileType(self.board[x][y])

    def at_(self, x, y):
        return self.board[x][y]

    def __init__(self, board=None, rows=8, cols=8):
        if board is not None:
            self.board = board
        else:
            self.board = np.full((rows, cols), TileType.EMPTY,
                                 dtype=np.dtype('uint8'))
        self.flips = []
        self.moves = []

    def get_score(self, player):
        # Determine the score by counting the tiles
        score = 0
        for x in np.nditer(self.board):
            if x == player.tiletype:
                score += 1
        return score

    def get_score_heuristic(self, player):
        # Determine the score by counting the tiles
        score = 0
        for x in np.nditer(self.board):
            if x == player.tiletype:
                score += 1
        return score

    def get_score_heuristic_(self, tiletype):
        # Determine the score by counting the tiles
        score = 0
        # for x in np.nditer(self.board):
        #     if x == tiletype:
        #         score += 1
        for x in range(self.rows):
            for y in range(self.cols):
                if self.board[x][y] == tiletype:
                    score += self.SCOREBOARD[x][y]

        return score

    def _check_bounds(self, x, y):
        # Check whether the x,y coords are inside the board
        return x >= 0 and y >= 0 and x < self.cols and y < self.rows

    def _check_move_legality(self, move):
        # Check the bounds and whether the cell is not occupied
        return (self._check_bounds(move.x, move.y) and
                self.board[move.x][move.y] == TileType.EMPTY)

    def _check_move_legality_(self, x, y):
        # Check the bounds and whether the cell is not occupied
        return (self._check_bounds(x, y) and
                self.board[x][y] == TileType.EMPTY)

    def _check_corner(self, x, y):
        # Check whether the x,y coords are on the corner of the board
        return ((x == 0 and y == 0) or
                (x == 0 and y == self.rows - 1) or
                (x == self.cols - 1 and y == 0) or
                (x == self.cols - 1 and y == self.rows - 1))

    def check_move(self, move):
        # Do a basic check on whether the move is legal
        # then calculate the flips and return the flipped pieces
        # print(move.x, move.y, move.player.tiletype)
        if not self._check_move_legality(move):
            return None
        # Starting the simulation of the move
        # First, we add the temp tile to the board
        self.board[move.x][move.y] = move.player.tiletype
        # Then, for each possible direction we calculate captured tiles (flips)
        flips = []
        for xdir, ydir in self.MOVE_DIRECTIONS:
            x, y = move.x + xdir, move.y + ydir
            if self._check_bounds(x, y) and self.at_(x, y) == move.player.tiletype.inverse:
                # The neighbour has another color
                # Since we checked the x,y we might as well finish this iteration
                # It leads to worse looking code but better performance
                x, y = x + xdir, y + ydir
                # We won't have our color outside of the board
                if not self._check_bounds(x, y):
                    continue  # The for loop
                while self._check_bounds(x, y) and self.at_(x, y) == move.player.tiletype.inverse:
                    x, y = x + xdir, y + ydir
                    if not self._check_bounds(x, y):
                        break  # The while loop
                if not self._check_bounds(x, y):
                    continue  # The for loop
                # We have to check it because blank tiles exist
                if self.at_(x, y) == move.player.tiletype:
                    # We now know that we have some tiles to flip
                    # We can move backwards and add them to the list
                    # Our current tile has the same color, so substract 1st
                    while True:
                        x, y = x - xdir, y - ydir
                        # We have to have a while true/break
                        # to avoid adding the original tile
                        if x == move.x and y == move.y:
                            break
                        flips.append([x, y])
        # Revert the cell to the original state
        self.board[move.x][move.y] = TileType.EMPTY
        # self._print_board()
        # print(flips, end='\n\n\n')
        return flips  # return flips, check for None/empty list on the other side

    def _print_board(self):
        for x in range(self.board.shape[0]):
            for y in range(self.board.shape[1]):
                print(self.board[x][y], end='')
            print()

    def check_move_(self, move_x, move_y, tiletype):
        # Do a basic check on whether the move is legal
        # then calculate the flips and return the flipped pieces
        if not self._check_move_legality_(move_x, move_y):
            return None
        # Starting the simulation of the move
        # First, we add the temp tile to the board
        self.board[move_x][move_y] = tiletype
        # Then, for each possible direction we calculate captured tiles (flips)
        flips = []
        for xdir, ydir in self.MOVE_DIRECTIONS:
            x, y = move_x + xdir, move_y + ydir
            if self._check_bounds(x, y) and self.board[x][y] == tiletype.inverse:
                # The neighbour has another color
                # Since we checked the x,y we might as well finish this iteration
                # It leads to worse looking code but better performance
                x, y = x + xdir, y + ydir
                # We won't have our color outside of the board
                if not self._check_bounds(x, y):
                    continue  # The for loop
                while self._check_bounds(x, y) and self.board[x][y] == tiletype.inverse:
                    x, y = x + xdir, y + ydir
                    if not self._check_bounds(x, y):
                        break  # The while loop
                if not self._check_bounds(x, y):
                    continue  # The for loop
                # We have to check it because blank tiles exist
                if self.board[x][y] == tiletype:
                    # We now know that we have some tiles to flip
                    # We can move backwards and add them to the list
                    # Our current tile has the same color, so substract 1st
                    while True:
                        x, y = x - xdir, y - ydir
                        # We have to have a while true/break
                        # to avoid adding the original tile
                        if x == move_x and y == move_y:
                            break
                        flips.append([x, y])
        # Revert the cell to the original state
        self.board[move_x][move_y] = TileType.EMPTY
        return flips  # return flips, check for None/empty list on the other side

    def get_legal_moves(self, player):
        legal_moves = []
        for x in range(self.cols):
            for y in range(self.rows):
                temp_move = Move(player, x, y)
                if self.check_move(temp_move):
                    legal_moves.append(temp_move)
        return legal_moves

    def get_legal_moves_pos(self, player):
        pos = []
        for move in self.get_legal_moves(player):
            pos.append((move.x, move.y))
        return pos

    def get_legal_moves_pos_(self, tiletype):
        pos = []
        for x in range(self.cols):
            for y in range(self.rows):
                if self.check_move_(x, y, tiletype):
                    pos.append([x, y])
        return pos

    def reset_board(self):
        self.board.fill(TileType.EMPTY)
        # The game starts with 4 diagonally placed tiles
        cols, rows = self.cols, self.rows
        self.board[cols // 2][rows // 2] = TileType.WHITE
        self.board[cols // 2 - 1][rows // 2 - 1] = TileType.WHITE

        self.board[cols // 2 - 1][rows // 2] = TileType.BLACK
        self.board[cols // 2][rows // 2 - 1] = TileType.BLACK

    def calc_move(self, move):

        flips = self.check_move(move)
        if not flips:
            return None

        self.board[move.x][move.y] = move.player.tiletype

        for x, y in flips:
            self.board[x][y] = move.player.tiletype

        self.flips.append(flips)
        self.moves.append([move.x, move.y])

        return flips

    def calc_move_(self, x, y, tiletype):
        flips = self.check_move_(x, y, tiletype)
        if not flips:
            return None

        self.board[x][y] = tiletype
        self.moves.append([x, y])

        for x, y in flips:
            self.board[x][y] = tiletype
        self.flips.append(flips)

        # print(x, y, tiletype, flips, 'self', self.flips)
        return flips

    def undo_last(self):
        # print(self.flips, 'moves', self.moves)
        last_flips = self.flips.pop()
        last_x, last_y = self.moves.pop()
        # print('UNDOING', last_flips, 'X:', last_x, 'Y', last_y)
        # self._print_board()
        # print('AFTER')
        # self._print_board()
        # print(end='\n\n\n')
        self.board[last_x][last_y] = TileType.EMPTY
        for x, y in last_flips:
            self.board[x][y] = TileType(self.board[x][y]).inverse


class Game:
    COLOR_TEXT = (0, 0, 0)
    COLOR_GRIDLINE = (0, 0, 0)
    COLOR_TEXT_BG_1 = (255, 255, 255)
    COLOR_TEXT_BG_2 = (0, 100, 0)
    COLOR_HINT = (174, 94, 0)

    def _board_to_pixels(self, x, y):
        return (self.margin_x + x * self.square_size + int(self.square_size / 2),
                self.margin_y + y * self.square_size + int(self.square_size / 2))

    def __init__(self, **kwargs):
        # BOARD
        self.board = Board(rows=kwargs.get('board_rows', 8),
                           cols=kwargs.get('board_cols', 8))
        # WINDOW
        self.fps = kwargs.get('fps', 30)

        self.square_size = kwargs.get('square_size', 90)

        self.margin_x = kwargs.get('margin_x', 15)
        self.margin_y = kwargs.get('margin_y', 15)
        self.x_line_widths = ([self.margin_x] +
                              [1] * (self.board.cols - 1) +
                              [self.margin_x])
        self.x_line_offsets = ([0.5 * self.margin_x] +
                               [self.margin_x] * (self.board.cols - 1) +
                               [1.5 * self.margin_x + 1])

        self.y_line_widths = ([self.margin_y] +
                              [1] * (self.board.rows - 1) +
                              [self.margin_y])
        self.y_line_offsets = ([0.5 * self.margin_y] +
                               [self.margin_y] * (self.board.rows - 1) +
                               [1.5 * self.margin_y])

        self.wheight = kwargs.get('wheight',
                                  (self.square_size *
                                   self.board.rows +
                                   2 * self.margin_y))
        self.wwidth = kwargs.get('wwidth',
                                 (self.square_size *
                                  self.board.cols + 2 *
                                  self.margin_x + 250))

        self.anim_speed = max(0, min(kwargs.get('anim_speed', 15), 100))

        # COLORS
        self.color_text_bg_1 = kwargs.get(
            'color_text_bg_1', self.COLOR_TEXT_BG_1)
        self.color_gridline = kwargs.get(
            'color_gridline', self.COLOR_GRIDLINE)
        self.color_text_bg_2 = kwargs.get(
            'color_text_bg_2', self.COLOR_TEXT_BG_2)
        self.color_text = kwargs.get('color_text', self.COLOR_TEXT)
        self.color_hint = kwargs.get('color_hint', self.COLOR_HINT)
        # STARTING PYGAME
        pygame.init()
        self.clock = pygame.time.Clock()
        self.displaysurf = pygame.display.set_mode(
            (self.wwidth, self.wheight), pygame.DOUBLEBUF)
        pygame.display.set_caption('Reverpy')
        self.font = pygame.font.Font('res/font.ttf', 16)
        self.font_big = pygame.font.Font('res/font.ttf', 32)
        # Stretch the picture to fit the board
        board_img = pygame.transform.smoothscale(
            pygame.image.load('res/board.png'),
            (1 + self.board.cols * self.square_size,
             self.board.rows * self.square_size))
        board_img_rect = board_img.get_rect()
        board_img_rect.topleft = (self.margin_x, self.margin_y)
        self.bg_img = pygame.transform.smoothscale(
            pygame.image.load('res/bg.png'), (self.wwidth, self.wheight))
        self.bg_img_pure = copy.copy(self.bg_img)
        # Merging the BG with the board
        self.bg_img.blit(board_img, board_img_rect)
        self.to_draw = []
        self.to_press = []

    def start(self):
        while True:
            if not self._game_loop():
                break

    def _info_zone_middle(self):
        x_start = 2 * self.margin_x + self.board.cols * self.square_size
        y_end = 2 * self.margin_y + self.board.rows * self.square_size
        return (x_start + (self.wwidth - x_start) / 2, y_end / 2)

    def _handle_processing(self):
        if self.current_turn == self.computer:
            self.check_quit()
            self._update_draws()
            for event in pygame.event.get():
                if event.type == MOUSEBUTTONUP:
                    mousex, mousey = event.pos
                    for rect, action in self.to_press:
                        if rect.collidepoint((mousex, mousey)):
                            action()
            pygame.display.update()
            self.clock.tick(self.fps)
        else:
            move_xy = None
            while not move_xy:
                self.check_quit()
                self._update_draws()
                for event in pygame.event.get():
                    if event.type == MOUSEBUTTONUP:
                        mousex, mousey = event.pos
                        for rect, action in self.to_press:
                            if rect.collidepoint((mousex, mousey)):
                                action()
                        move_xy = self._get_clicked_space(mousex, mousey)
                        if move_xy and not self.board.check_move(Move(self.current_turn,
                                                                      *move_xy)):
                            move_xy = None
                pygame.display.update()
                self.clock.tick(self.fps)
            return move_xy

    def _update_draws(self):
        for func, params in self.to_draw:
            func(*params)
        pygame.display.update()
        self.clock.tick(self.fps)

    def _change_ai_naive(self):
        self.computer.update_aitype(AiType.NAIVE)
        self._update_draws()

    def _change_ai_minmax(self):
        self.computer.update_aitype(AiType.MINMAX)
        self._update_draws()

    def _change_ai_abeta(self):
        self.computer.update_aitype(AiType.ABETA)
        self._update_draws()

    def _game_loop(self):
        self.board.reset_board()
        self.to_draw = []
        self.to_press = []

        self.displaysurf.blit(self.bg_img_pure, self.bg_img_pure.get_rect())

        self.person = Player(type=PlayerType.PERSON,
                             tiletype=self._select_player_tiletype(),
                             opponent=None, aitype=AiType.PERSON)
        self.computer = Player(type=PlayerType.COMPUTER,
                               tiletype=self.person.tiletype.inverse,
                               opponent=self.person, aitype=AiType.NAIVE)

        if self.person.tiletype == TileType.BLACK:
            self.current_turn = self.person
        else:
            self.current_turn = self.computer

        self._draw_board()

        # Making buttons
        mid_x, mid_y = self._info_zone_middle()
        self._make_button(
            'RESTART', mid_x, mid_y + 300,
            width=125, height=50,
            color_rect_active=(255, 0, 0), color_rect_inactive=(0, 0, 0),
            color_text=(255, 255, 255), action=self.start, draw_func=self._draw_button)

        self._make_button(
            'NAIVE', mid_x, mid_y,
            width=75, height=35,
            color_rect_active=(255, 0, 0), color_rect_inactive=(0, 0, 0),
            color_text=(255, 255, 255), action=self._change_ai_naive, draw_func=self._draw_button_naive)

        self._make_button(
            'MINMAX', mid_x, mid_y + 40,
            width=75, height=35,
            color_rect_active=(255, 0, 0), color_rect_inactive=(0, 0, 0),
            color_text=(255, 255, 255), action=self._change_ai_minmax, draw_func=self._draw_button_minmax)

        self._make_button(
            'ABETA', mid_x, mid_y + 80,
            width=75, height=35,
            color_rect_active=(255, 0, 0), color_rect_inactive=(0, 0, 0),
            color_text=(255, 255, 255), action=self._change_ai_abeta, draw_func=self._draw_button_abeta)

        while True:
            if self.current_turn == self.person:
                if not self.board.get_legal_moves(self.current_turn):
                    break

                self._draw_board()
                self._draw_info(self.person, self.computer)
                self._update_draws()

                response = self._handle_processing()
                if not response:
                    return True
                chosen_move = Move(self.current_turn, *response)
                flips = self.board.calc_move(chosen_move)
                self._animate_tiles(flips, chosen_move)
                self.current_turn = self.computer
            else:
                if not self.board.get_legal_moves(self.current_turn):
                    break

                self._draw_board()
                self._draw_info(self.person, self.computer)
                self._update_draws()
                self._handle_processing()
                comp_move = self.computer.move(self.board)
                flips = self.board.calc_move(comp_move)
                self._animate_tiles(flips, comp_move)
                self.current_turn = self.person

        # Display the final score.
        self._draw_board()
        player_score, computer_score = (self.board.get_score(self.person),
                                        self.board.get_score(self.computer))
        # Reset the bg
        self.displaysurf.blit(self.bg_img_pure, self.bg_img_pure.get_rect())
        # Determine the text of the message to display.
        if player_score > computer_score:
            text = 'You beat the computer by {0} points! Congratulations!'.format(
                player_score - computer_score)
        elif player_score < computer_score:
            text = 'You lost. The computer beat you by {0} points.'.format(
                computer_score - player_score)
        else:
            text = 'The game was a tie!'

        text_surf = self.font.render(
            text, True, self.color_text, self.color_text_bg_1)
        text_rect = text_surf.get_rect()
        text_rect.center = (int(self.wwidth / 2), int(self.wheight / 2))
        self.displaysurf.blit(text_surf, text_rect)

        # Display the "Play again?" text with Yes and No buttons.
        text2_surf = self.font_big.render(
            'Play again?', True, self.color_text, self.color_text_bg_1)
        text2_rect = text2_surf.get_rect()
        text2_rect.center = (int(self.wwidth / 2), int(self.wheight / 2) + 50)

        # Make "Yes" button.
        yes_surf = self.font_big.render(
            'Yes', True, self.color_text, self.color_text_bg_1)
        yes_rect = yes_surf.get_rect()
        yes_rect.center = (int(self.wwidth / 2) - 60,
                           int(self.wheight / 2) + 90)

        # Make "No" button.
        no_surf = self.font_big.render(
            'No', True, self.color_text, self.color_text_bg_1)
        no_rect = no_surf.get_rect()
        no_rect.center = (int(self.wwidth / 2) + 60,
                          int(self.wheight / 2) + 90)

        while True:
            self.check_quit()
            for event in pygame.event.get():
                if event.type == MOUSEBUTTONUP:
                    mousex, mousey = event.pos
                    if yes_rect.collidepoint((mousex, mousey)):
                        return True
                    elif no_rect.collidepoint((mousex, mousey)):
                        return False
            self.displaysurf.blit(text_surf, text_rect)
            self.displaysurf.blit(text2_surf, text2_rect)
            self.displaysurf.blit(yes_surf, yes_rect)
            self.displaysurf.blit(no_surf, no_rect)
            pygame.display.update()
            self.clock.tick(self.fps)

    def _animate_tiles(self, flips, last_turn):
        # Draw the last tile
        self._draw_tile(last_turn.x, last_turn.y,
                        last_turn.player.tiletype.color, update=True)

        for rgb_vals in range(0, 255, int(self.anim_speed * 2.55)):
            rgb_vals = max(0, min(255, rgb_vals))

            if last_turn.player.tiletype == TileType.WHITE:
                color = tuple([rgb_vals] * 3)  # from 0 to 255
            else:
                color = tuple([255 - rgb_vals] * 3)  # from 255 to 0

            for x, y in flips:
                self._draw_tile(x, y, color)

            pygame.display.update()
            self.clock.tick(self.fps)
            self.check_quit()

    def _draw_tile(self, x, y, color, update=False, scale=1):
        x, y = self._board_to_pixels(x, y)
        pygame.draw.circle(self.displaysurf, color,
                           (x, y), int(scale * self.square_size // 2) - 4)
        if update:
            pygame.display.update()

    def _make_button(self, msg, x, y, width, height,
                     color_rect_active, color_rect_inactive,
                     color_text, action, draw_func):
        params = [msg, x, y, width, height,
                  color_rect_active, color_rect_inactive,
                  color_text]
        rect = draw_func(*params)
        self.to_press.append([rect, action])
        self.to_draw.append([draw_func, params])

    def _draw_button(self, msg, x, y, width, height,
                     color_rect_active, color_rect_inactive,
                     color_text):
        mouse = pygame.mouse.get_pos()
        x_start, x_end, y_start, y_end = (x - width / 2,
                                          x + width / 2,
                                          y - height / 2,
                                          y + height / 2)
        if x_start < mouse[0] < x_end and y_start < mouse[1] < y_end:
            pygame.draw.rect(self.displaysurf, color_rect_active,
                             (x_start, y_start, width, height))
        else:
            pygame.draw.rect(self.displaysurf, color_rect_inactive,
                             (x_start, y_start, width, height))

        surf = self.font.render(msg, True, color_text)
        rect = surf.get_rect()
        rect.center = (x, y)
        self.displaysurf.blit(surf, rect)
        return rect

    def _draw_button_naive(self, msg, x, y, width, height,
                           color_rect_active, color_rect_inactive,
                           color_text):
        x_start, y_start = (x - width / 2, y - height / 2)
        if self.computer.aitype == AiType.NAIVE:
            pygame.draw.rect(self.displaysurf, color_rect_active,
                             (x_start, y_start, width, height))
        else:
            pygame.draw.rect(self.displaysurf, color_rect_inactive,
                             (x_start, y_start, width, height))

        surf = self.font.render(msg, True, color_text)
        rect = surf.get_rect()
        rect.center = (x, y)
        self.displaysurf.blit(surf, rect)
        return rect

    def _draw_button_minmax(self, msg, x, y, width, height,
                            color_rect_active, color_rect_inactive,
                            color_text):
        x_start, y_start = (x - width / 2, y - height / 2)
        if self.computer.aitype == AiType.MINMAX:
            pygame.draw.rect(self.displaysurf, color_rect_active,
                             (x_start, y_start, width, height))
        else:
            pygame.draw.rect(self.displaysurf, color_rect_inactive,
                             (x_start, y_start, width, height))

        surf = self.font.render(msg, True, color_text)
        rect = surf.get_rect()
        rect.center = (x, y)
        self.displaysurf.blit(surf, rect)
        return rect

    def _draw_button_abeta(self, msg, x, y, width, height,
                           color_rect_active, color_rect_inactive,
                           color_text):
        x_start, y_start = (x - width / 2, y - height / 2)
        if self.computer.aitype == AiType.ABETA:
            pygame.draw.rect(self.displaysurf, color_rect_active,
                             (x_start, y_start, width, height))
        else:
            pygame.draw.rect(self.displaysurf, color_rect_inactive,
                             (x_start, y_start, width, height))

        surf = self.font.render(msg, True, color_text)
        rect = surf.get_rect()
        rect.center = (x, y)
        self.displaysurf.blit(surf, rect)
        return rect

    def _draw_board(self):
        # Draw the bg
        # self._update_blit()
        self.displaysurf.blit(self.bg_img, self.bg_img.get_rect())

        # DRAW THE GRID
        # Vertical lines
        for x in range(self.board.cols + 1):
            x_pos = self.x_line_offsets[x] + (x * self.square_size)
            y_start = self.margin_y
            y_end = self.margin_y + (self.board.cols * self.square_size)
            pygame.draw.line(self.displaysurf, self.color_gridline,
                             (x_pos, y_start), (x_pos, y_end), self.x_line_widths[x])

        # Horizontal lines
        for y in range(self.board.rows + 1):
            x_start = 0
            x_end = 2 * self.margin_x + (self.board.cols * self.square_size)
            y_pos = self.y_line_offsets[y] + (y * self.square_size)
            pygame.draw.line(self.displaysurf, self.color_gridline,
                             (x_start, y_pos), (x_end, y_pos), self.y_line_widths[y])

        # DRAW TILES
        for x in range(self.board.cols):
            for y in range(self.board.rows):
                if self.board.at(x, y).placed:
                    self._draw_tile(x, y, self.board.at(x, y).color)

        # DRAW HINTS
        if self.current_turn.person:
            hints = self.board.get_legal_moves_pos(self.current_turn)
            for x, y in hints:
                self._draw_tile(x, y, self.COLOR_HINT, scale=0.25)

    def _draw_info(self, person, computer):
        # Draws scores and whose turn it is at the bottom of the screen.
        person_score = self.board.get_score(person)
        computer_score = self.board.get_score(computer)
        x_mid, y_mid = self._info_zone_middle()

        cturn_surf = self.font_big.render('{}\'s'.format(self.current_turn.type.name.title()),
                                          True, self.color_gridline)
        cturn_rect = cturn_surf.get_rect()
        cturn_rect.center = (x_mid, y_mid - 150)

        turn_surf = self.font_big.render('Turn'.format(self.current_turn.type.name.title()),
                                         True, self.color_gridline)
        turn_rect = turn_surf.get_rect()
        turn_rect.center = (x_mid, y_mid - 120)

        computer_score_surf = self.font.render('Computer Score: {}'.format(computer_score),
                                               True, self.color_gridline)
        computer_score_rect = computer_score_surf.get_rect()
        computer_score_rect.center = (x_mid, y_mid - 50)

        player_score_surf = self.font.render('Player Score: {}'
                                             .format(person_score),
                                             True, self.color_gridline)
        player_score_rect = player_score_surf.get_rect()
        player_score_rect.center = (x_mid, y_mid - 75)

        self.displaysurf.blit(player_score_surf, player_score_rect)
        self.displaysurf.blit(computer_score_surf, computer_score_rect)
        self.displaysurf.blit(cturn_surf, cturn_rect)
        self.displaysurf.blit(turn_surf, turn_rect)

    def _get_clicked_space(self, mousex, mousey):
        # Return a tuple of two integers of the board space coordinates where
        # the mouse was clicked. (Or returns None not in any space.)
        for x in range(self.board.cols):
            for y in range(self.board.rows):
                if (mousex > x * self.square_size + self.margin_x and
                    mousex < (x + 1) * self.square_size + self.margin_x and
                    mousey > y * self.square_size + self.margin_y and
                        mousey < (y + 1) * self.square_size + self.margin_y):
                    return (x, y)
        return None

    def _select_player_tiletype(self):
        text_surf = self.font.render(
            'Do you want to play as white or black?', True, self.color_text, self.color_text_bg_1)
        text_rect = text_surf.get_rect()
        text_rect.center = (int(self.wwidth / 2), int(self.wheight / 2))

        white_surf = self.font_big.render(
            'White', True, self.color_text, self.color_text_bg_1)
        white_rect = white_surf.get_rect()
        white_rect.center = (int(self.wwidth / 2) - 60,
                             int(self.wheight / 2) + 40)

        black_surf = self.font_big.render(
            'Black', True, self.color_text, self.color_text_bg_1)
        black_rect = black_surf.get_rect()
        black_rect.center = (int(self.wwidth / 2) + 60,
                             int(self.wheight / 2) + 40)

        while True:
            # Keep looping until the player has clicked on a color.
            self.check_quit()
            for event in pygame.event.get():
                if event.type == MOUSEBUTTONUP:
                    mousex, mousey = event.pos
                    if white_rect.collidepoint((mousex, mousey)):
                        return TileType.WHITE
                    elif black_rect.collidepoint((mousex, mousey)):
                        return TileType.BLACK

            self.displaysurf.blit(text_surf, text_rect)
            self.displaysurf.blit(white_surf, white_rect)
            self.displaysurf.blit(black_surf, black_rect)
            pygame.display.update()
            self.clock.tick(self.fps)

    def check_quit(self):
        for event in pygame.event.get((QUIT)):  # event handling loop
            if event.type == QUIT:
                pygame.quit()
                sys.exit()


if __name__ == '__main__':
    sys.setrecursionlimit(10000000)
    reverpy = Game()
    reverpy.start()
