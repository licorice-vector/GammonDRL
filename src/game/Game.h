#ifndef GAME_H
#define GAME_H

#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <cassert>
#include <algorithm>
#include <memory>
#include <set>
#include <stack>

#include "../player/Player.h"

namespace Backgammon {
    #define WHITE 0
    #define BLACK 1
    #define BAR 24
    #define OUT 25
    #define BOARD_SIZE 24

    enum Outcome { 
        WON_SINGLE_GAME, 
        WON_GAMMON, 
        WON_BACKGAMMON, 
        LOST_SINGLE_GAME, 
        LOST_GAMMON, 
        LOST_BACKGAMMON
    };
    
    class Player;
    class CheckerMove;
    class State;
    class Dice;
    class Game;

    typedef std::vector<CheckerMove> Move;
    typedef std::vector<Move> Moves;
    typedef std::vector<int> Deltas;
    typedef std::vector<std::pair<int, CheckerMove>> Undo;
    
    class CheckerMove {
    public:
        int from;
        int to;
        CheckerMove(int from, int to) : from(from), to(to) {}
    };

    bool operator<(const CheckerMove& a, const CheckerMove& b);
    bool operator==(const CheckerMove& a, const CheckerMove& b);

    class State {
    public:
        int turn;
        std::array<std::array<int, 6>, 2> home = {
            std::array<int, 6>{5, 4, 3, 2, 1, 0},
            std::array<int, 6>{18, 19, 20, 21, 22, 23}
        };
        std::array<std::array<int, 26>, 2> on;
        std::stack<Undo> made;
        State();
        int compute_pip(int player) const;
        bool race() const;
        void make_move(Move move);
        void undo_move();
        Moves get_moves(Deltas deltas);
        void find_moves(const Deltas& deltas, int index, Move& move, Moves& moves);
        bool can_bear_off();
        bool can_bear_off(int from, int delta);
        Outcome outcome(int player) const;
        void show() const;
    };

    class Dice {
    public:
        int first;
        int second;
        static std::random_device rd;
        static std::mt19937 rng;
        std::uniform_int_distribution<> uniform;
        bool first_throw;
        Dice();
        void roll();
        Deltas get_deltas();
    };

    class Game {
    public:
        std::array<int, 2> points;
        State state;
        Dice dice;
        std::array<std::shared_ptr<Player>, 2> players;
        Game(std::shared_ptr<Player> white, std::shared_ptr<Player> black);
        void play_turn();
        void play();
    };
}

#endif
