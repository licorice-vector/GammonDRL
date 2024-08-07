#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <cassert>
#include <algorithm>
#include <memory>

#include "Game.h"

namespace Backgammon {

    bool operator<(const CheckerMove& a, const CheckerMove& b) {
        return a.from == b.from ? a.to < b.to : a.from < b.from;
    }

    bool operator==(const CheckerMove& a, const CheckerMove& b) {
        return a.from == b.from && a.to == b.to;
    }

    State::State() {
        // beginning state
        // turn is not decided yet
        turn = -1;
        on[WHITE].fill(0);
        on[BLACK].fill(0);
        on[BLACK][0] = 2;
        on[WHITE][5] = 5;
        on[WHITE][7] = 3;
        on[BLACK][11] = 5;
        on[WHITE][12] = 5;
        on[BLACK][16] = 3;
        on[BLACK][18] = 5;
        on[WHITE][23] = 2;
    }

    bool State::can_bear_off() {
        int cnt = 0;
        for (auto point : home[turn]) {
            cnt += on[turn][point];
        }
        return cnt + on[turn][OUT] == 15;
    }
    
    void State::make_move(Move move) {
        Undo undo;
        for (auto checker_move : move) {
            auto [from, to] = checker_move;
            if (to != OUT && on[!turn][to]) {
                on[!turn][BAR]++;
                on[!turn][to]--;
                undo.push_back({!turn, CheckerMove(to, BAR)});
            }
            on[turn][from]--;
            on[turn][to]++;
            undo.push_back({turn, checker_move});
        }
        made.push(undo);
    }

    void State::undo_move() {
        auto undo = made.top();
        made.pop();
        for (auto [turn, checker_move] : undo) {
            auto [from, to] = checker_move;
            on[turn][to]--;
            on[turn][from]++;
        }
    }

    Move State::get_move(int delta) {
        int direction = turn == WHITE ? -1 : 1;
        Move move;
        if (on[turn][BAR]) {
            int to = (turn == WHITE ? 24 : -1) + direction * delta;
            if (on[!turn][to] <= 1) {
                move.insert(CheckerMove(BAR, to));
            }
            return move;
        }
        if (can_bear_off()) {
            int bear_off_point = (turn == WHITE ? -1 : 24) + -1 * direction * delta;
            if (on[turn][bear_off_point]) {
                move.insert(CheckerMove(bear_off_point, OUT));
            }
        }
        for (int from = 0; from < BOARD_SIZE; from++) {
            int to = from + direction * delta;
            if (on[turn][from] && 0 <= to && to < BOARD_SIZE && on[!turn][to] <= 1) {
                move.insert(CheckerMove(from, to));
            }
        }
        return move;
    }

    bool State::no_moves(const Deltas& deltas, int index) {
        while (index < (int)deltas.size()) {
            if (get_move(deltas[index]).size()) {
                return false;
            }
            index++;
        }
        return true;
    }

    Moves State::get_moves(Deltas deltas) {
        std::set<Move> seen;
        Move move;
        Moves moves;
        get_moves(deltas, 0, seen, move, moves);
        if (deltas[0] != deltas[1]) {
            std::swap(deltas[0], deltas[1]);
            seen.clear();
            get_moves(deltas, 0, seen, move, moves);
        }
        auto it = std::max_element(moves.begin(), moves.end(), [] (const Move& a, const Move& b) {
            return a.size() < b.size();
        });
        if (it != moves.end()) {
            auto mx = it->size();
            moves.erase(std::remove_if(moves.begin(), moves.end(), [mx] (const Move& move) {
                return move.size() != mx;
            }), moves.end());
            std::sort(moves.begin(), moves.end());
            moves.erase(std::unique(moves.begin(), moves.end()), moves.end());
        }
        return moves;
    }

    void State::get_moves(const Deltas& deltas, int index, std::set<Move>& seen, Move& move, Moves& moves) {
        if (!move.empty() && seen.count(move)) {
            return;
        }
        seen.insert(move);
        if (index == (int)deltas.size()) {
            if (!move.empty()) {
                moves.push_back(move);
            }
            return;
        }
        if (can_bear_off() && no_moves(deltas, index)) {
            Move to_be_erased;
            for (auto point : home[turn]) {
                while (on[turn][point] && index < (int)deltas.size()) {
                    CheckerMove checker_move(point, OUT);
                    move.insert(checker_move);
                    to_be_erased.insert(checker_move);
                    make_move({checker_move});
                    index++;
                }
            }
            if (!move.empty()) {
                moves.push_back(move);
            }
            for (auto checker_move : to_be_erased) {
                undo_move();
                move.erase(move.find(checker_move));
            }
            return;
        }
        if (no_moves(deltas, index)) {
            if (!move.empty()) {
                moves.push_back(move);
            }
            return;
        }
        int delta = deltas[index];
        Move delta_move = get_move(delta);
        if (delta_move.empty()) {
            get_moves(deltas, index + 1, seen, move, moves);
            return;
        }
        for (auto checker_move : delta_move) {
            make_move({checker_move});
            move.insert(checker_move);
            get_moves(deltas, index + 1, seen, move, moves);
            move.erase(move.find(checker_move));
            undo_move();
        }
    }

    Outcome State::outcome(int player) const {
        if (on[player][OUT] == 15) {
            if (on[!player][OUT]) {
                return Outcome::WON_SINGLE_GAME;
            }
            if (on[!player][BAR]) {
                return Outcome::WON_BACKGAMMON;
            }
            for (auto point : home[player]) {
                if (on[!player][point]) {
                    return Outcome::WON_BACKGAMMON;
                }
            }
            return Outcome::WON_GAMMON;
        }
        if (on[player][OUT]) {
            return Outcome::LOST_SINGLE_GAME;
        }
        if (on[player][BAR]) {
            return Outcome::LOST_BACKGAMMON;
        }
        for (auto point : home[!player]) {
            if (on[player][point]) {
                return Outcome::LOST_BACKGAMMON;
            }
        }
        return Outcome::LOST_GAMMON;
    }

    void State::show() const {
        /*
            |-----------------------------------|-----|-----------------------------------|-----|
            | 013 | 014 | 015 | 016 | 017 | 018 |     | 019 | 020 | 021 | 022 | 023 | 024 | OUT |
            |-----------------------------------|-----|-----------------------------------|-----|
            |  #  |     |     |     |  +  |     |     |  +  |     |     |     |     |  #  |     |
            |-----------------------------------|-BAR-|-----------------------------------|-----|
            |  +  |     |     |     |  #  |     |     |  #  |     |     |     |     |  +  |     |
            |-----------------------------------|-----|-----------------------------------|-----|
            | 012 | 011 | 010 | 009 | 008 | 007 |     | 006 | 005 | 004 | 003 | 002 | 001 | OUT |
            |-----------------------------------|-----|-----------------------------------|-----|
        */
        std::string white = "#";
        std::string black = "O";
        std::cout << "|-----------------------------------|-----|-----------------------------------|-----|" << std::endl;
        std::cout << "| 013 | 014 | 015 | 016 | 017 | 018 |     | 019 | 020 | 021 | 022 | 023 | 024 | OUT |" << std::endl;
        std::cout << "|-----------------------------------|-----|-----------------------------------|-----|" << std::endl;
        for (int row = 0; row < 15; row++) {
            std::cout << "|";
            for (int point = 12; point < BOARD_SIZE; point++) {
                std::cout << "  " << (on[WHITE][point] > row ? white : on[BLACK][point] > row ? black : " ") << "  |";
                if (point == 17) {
                    std::cout << "  " + std::string(on[WHITE][BAR] > 14 - row ? white : " ") + "  |";
                }
            }
            std::cout << "  " << (on[BLACK][OUT] > row ? black : " ") << "  |" << std::endl;
        }
        std::cout << "|-----------------------------------|-BAR-|-----------------------------------|-----|" << std::endl;
        for (int row = 14; row >= 0; row--) {
            std::cout << "|";
            for (int point = 11; point >= 0; point--) {
                std::cout << "  " << (on[WHITE][point] > row ? white : on[BLACK][point] > row ? black : " ") << "  |";
                if (point == 6) {
                    std::cout << "  " + std::string(on[BLACK][BAR] > 14 - row ? black : " ") + "  |";
                }
            }
            std::cout << "  " << (on[WHITE][OUT] > row ? white : " ") << "  |" << std::endl;
        }
        std::cout << "|-----------------------------------|-----|-----------------------------------|-----|" << std::endl;
        std::cout << "| 012 | 011 | 010 | 009 | 008 | 007 |     | 006 | 005 | 004 | 003 | 002 | 001 | OUT |" << std::endl;
        std::cout << "|-----------------------------------|-----|-----------------------------------|-----|" << std::endl;
        std::cout << (turn == WHITE ? "White" : "Black") << "'s turn" << std::endl;
    }

    Dice::Dice() 
        : first(0),
        second(0),
        uniform(std::uniform_int_distribution<>(1, 6)),
        first_throw(true) {}
    
    void Dice::roll() {
        if (first_throw) {
            first_throw = false;
            return;
        }
        first = uniform(rng);
        second = uniform(rng);
    }

    Deltas Dice::get_deltas() {
        std::vector<int> deltas;
        if (first == second) {
            deltas = {first, second, first, second};
        } else {
            deltas = {first, second};
        }
        return deltas;
    }

    std::random_device Dice::rd = std::random_device();
    std::mt19937 Dice::rng = std::mt19937(rd());

    Game::Game(std::shared_ptr<Player> white, std::shared_ptr<Player> black) {
        points.fill(0);
        players[WHITE] = white;
        players[BLACK] = black;
    }

    void Game::play_turn() {
        dice.roll();
        //state.show();
        //std::cout << "Dice: " << dice.first << " " << dice.second << std::endl;
        Moves moves = state.get_moves(dice.get_deltas());
        if (moves.empty()) {
            players[state.turn]->no_moves(state);
            state.turn = !state.turn;
            return;
        }
        int index = players[state.turn]->choose_move(state, dice, moves);
        //std::cout << "Moved the following checkers (from, to):" << std::endl;
        //for (auto [from, to] : moves[index]) {
        //    std::cout << "(" << from + 1 << ", " << to + 1 << ")" << std::endl;
        //}
        state.make_move(moves[index]);
        state.turn = !state.turn;
    }

    void Game::play() {
        players[WHITE]->new_game();
        players[BLACK]->new_game();
        state = State();
        dice = Dice();
        do {
            dice.roll();
        } while (dice.first == dice.second);
        dice.first_throw = true;
        state.turn = dice.first < dice.second ? WHITE : BLACK;
        while (true) {
            play_turn();
            if (state.on[!state.turn][OUT] == 15) {
                players[WHITE]->game_over(state, WHITE);
                players[BLACK]->game_over(state, BLACK);
                //state.show();
                Outcome outcome = state.outcome(WHITE);
                std::string s;
                if (outcome == Outcome::WON_SINGLE_GAME) {
                    s = " won a single game";
                    points[WHITE] += 1;
                } else if (outcome == Outcome::WON_GAMMON) {
                    s = " won a gammon";
                    points[WHITE] += 2;
                } else if (outcome == Outcome::WON_BACKGAMMON) {
                    s = " won a backgammon";
                    points[WHITE] += 3;
                } else if (outcome == Outcome::LOST_SINGLE_GAME) {
                    s = " lost a single game";
                    points[BLACK] += 1;
                } else if (outcome == Outcome::LOST_GAMMON) {
                    s = " lost a gammon";
                    points[BLACK] += 2;
                } else if (outcome == Outcome::LOST_BACKGAMMON) {
                    s = " lost a backgammon";
                    points[BLACK] += 3;
                }
                //std::cout << "White " << s << std::endl;
                break;
            }
        }
    }
}
