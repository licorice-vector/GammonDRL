#include "Human.h"

namespace Backgammon {
    Human::Human(std::string name) : name(name) {}
    
    int Human::choose_move(const State& state, const Dice& dice, const Moves& moves) {
        if ((int)moves.size() <= 7) {
            std::cout << "Choose a move:" << std::endl;
            for (int i = 0; i < (int)moves.size(); i++) {
                std::cout << "[" << i + 1 << "]: ";
                for (auto [from, to] : moves[i]) {
                    std::cout << "(" << from + 1 << ", " << to + 1 << ") ";
                }
                std::cout << std::endl;
            }
            int index = -1;
            while (index == -1) {
                std::string s;
                std::cout << "> ";
                std::cin >> s;
                bool number = true;
                for (auto& c : s) {
                    if (!isdigit(c)) {
                        number = false;
                    }
                }
                if (!number) {
                    std::cout << s << " is not a number" << std::endl;
                }
                index = stoi(s);
                if (index <= 0 && index > (int)moves.size()) {
                    std::cout << index << " is not between 1 and " << (int)moves.size() << std::endl;
                    index = -1;
                }
            }
            index--;
            assert(0 <= index && index < (int)moves.size());
            return index;
        }
        int index = -1;
        Move move;
        while (index == -1) {
            std::cout << "Checker moves added:" << std::endl;
            for (auto [from, to] : move) {
                std::cout << "(" << from + 1 << ", " << to + 1 << ") ";
            }
            std::cout << std::endl;
            std::cout << "[0] add checker move" << std::endl;
            std::cout << "[1] play move" << std::endl;
            std::cout << "[2] redo" << std::endl;
            std::cout << "> ";
            std::string s;
            std::cin >> s;
            bool number = true;
            for (auto& c : s) {
                if (!isdigit(c)) {
                    number = false;
                }
            }
            if (!number) {
                std::cout << s << " is not a number" << std::endl;
                continue;
            }
            int action = stoi(s);
            if (action < 0 || action > 2) {
                std::cout << action << " is not 0, 1 or 2" << std::endl;
                continue;
            }
            if (action == 0) {
                std::cout << "[from] > ";
                std::cin >> s;
                number = true;
                for (auto& c : s) {
                    if (!isdigit(c)) {
                        number = false;
                    }
                    if (isalpha(c)) {
                        c = tolower(c);
                    }
                }
                if (!number && s != "bar" && s != "out") {
                    std::cout << s << " is not a number, BAR or OUT" << std::endl;
                    continue;
                }
                int from;
                if (number) {
                    from = stoi(s);
                } else if (s == "bar") {
                    from = BAR + 1;
                } else {
                    from = OUT + 1;
                }
                std::cout << "[to] > ";
                std::cin >> s;
                number = true;
                for (auto c : s) {
                    if (!isdigit(c)) {
                        number = false;
                    }
                    if (isalpha(c)) {
                        c = tolower(c);
                    }
                }
                if (!number && s != "bar" && s != "out") {
                    std::cout << s << " is not a number, BAR or OUT" << std::endl;
                    continue;
                }
                int to;
                if (number) {
                    to = stoi(s);
                } else if (s == "bar") {
                    to = BAR + 1;
                } else {
                    to = OUT + 1;
                }
                from--, to--;
                move.insert(CheckerMove(from, to));
            } else if (action == 1) {
                for (int i = 0; i < (int)moves.size(); i++) {
                    if (moves[i] == move) {
                        index = i;
                        break;
                    }
                }
                if (index == -1) {
                    std::cout << "Illegal move" << std::endl;
                    move.clear();
                }
            } else {
                move.clear();
            }
        }
        assert(0 <= index && index < (int)moves.size());
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
