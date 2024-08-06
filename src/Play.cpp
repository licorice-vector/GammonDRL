#include "./game/Game.h"
#include "./player/Player.h"
#include "./player/Human.h"
#include "./player/AI.h"

using namespace Backgammon;

int main() {

    std::vector<int> start = {300'000, 50'000};
    
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
    }
    if (start[BLACK]) {
        model[BLACK]->load(start_filename[BLACK]);
        std::cout << "Loaded weights for black from file: " << start_filename[BLACK] << std::endl;
    }

    Game game(
        std::make_shared<AI>("WHITE", model[WHITE]), 
        std::make_shared<AI>("BLACK", model[BLACK])
    );
    game.play();

    return 0;
}
