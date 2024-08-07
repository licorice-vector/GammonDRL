#include "./game/Game.h"
#include "./player/Player.h"
#include "./player/Human.h"
#include "./player/AI.h"

using namespace Backgammon;

int main() {

    std::vector<int> opponents = {
        0,
        50'000,
        100'000,
        150'000,
        200'000,
        250'000,
        300'000,
        400'000,
        600'000,
        800'000,
        1'000'000,
        1'200'000,
        1'400'000,
        1'600'000,
        1'800'000,
        2'000'000
    };
    std::vector<int> good_opponents = {
        150'000,
        600'000,
        800'000,
        1'000'000,
        1'400'000,
        1'600'000,
        1'800'000,
        2'000'000
    };
    for (int opponent : good_opponents) {
        std::vector<int> start = {opponent, 2'000'000};
        
        // Weight filenames
        std::vector<std::string> start_filename = {
            "weights/" + std::to_string(start[WHITE]) + "_games.csv",
            "weights/" + std::to_string(start[BLACK]) + "_games.csv"
        };

        // Model
        std::vector<std::shared_ptr<Model>> model = {
            std::make_shared<Model>(), 
            std::make_shared<Model>()
        };

        // Load model
        if (start[WHITE]) {
            model[WHITE]->load(start_filename[WHITE]);
            std::cout << "Loaded weights for white from file: " << start_filename[WHITE] << std::endl;
        } else {
            std::cout << "Using random weights for white" << std::endl;
        }

        if (start[BLACK]) {
            model[BLACK]->load(start_filename[BLACK]);
            std::cout << "Loaded weights for black from file: " << start_filename[BLACK] << std::endl;
        } else {
            std::cout << "Using random weights for black" << std::endl;
        }

        // Game
        Game game(
            //std::make_shared<Human>("WHITE"), 
            std::make_shared<AI>("WHITE", model[WHITE]), 
            std::make_shared<AI>("BLACK", model[BLACK])
        );

        int games = 10'000;
        for (int i = 1; i <= games; i++) {
            game.play();
        }

        std::cout << "White's points: " << game.points[WHITE] << std::endl;
        std::cout << "Black's points: " << game.points[BLACK] << std::endl;
    }

    return 0;
}
