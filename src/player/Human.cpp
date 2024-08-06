#include "Human.h"

namespace Backgammon {
    Human::Human(std::string name) : name(name) {}
    
    int Human::choose_move(const State& state, const Dice& dice, const Moves& moves) {
        std::cout << "Choose a move:" << std::endl;
        for (int i = 0; i < (int)moves.size(); i++) {
            std::cout << "[" << i + 1 << "]: ";
            for (auto [from, to] : moves[i]) {
                std::cout << "(" << from + 1 << ", " << to + 1 << ") ";
            }
            std::cout << std::endl;
        }
        int index;
        std::cin >> index;
        index--;
        assert(0 <= index && index < (int)moves.size());
        std::cout << std::endl;
        return index;
    }

    void Human::new_game() {}

    void Human::no_moves(const State& state) {
        std::cout << "No moves" << std::endl;
        std::cout << std::endl;
    }

    void Human::game_over(const State& state, int player) {
        Outcome outcome = state.outcome(player);
        std::string s;
        if (outcome == Outcome::WON_SINGLE_GAME) {
            s = " won a single game";
        } else if (outcome == Outcome::WON_GAMMON) {
            s = " won a gammon";
        } else if (outcome == Outcome::WON_BACKGAMMON) {
            s = " won a backgammon";
        } else if (outcome == Outcome::LOST_SINGLE_GAME) {
            s = " lost a single game";
        } else if (outcome == Outcome::LOST_GAMMON) {
            s = " lost a gammon";
        } else if (outcome == Outcome::LOST_BACKGAMMON) {
            s = " lost a backgammon";
        }
        std::cout << name << " " << s << std::endl;
    }
}
