#include "./game/Game.h"
#include "./player/Player.h"
#include "./player/Human.h"
#include "./player/Trainer.h"

using namespace Backgammon;

int main() {

    int start = 0;
    int end = 10'000'000;
    int hidden_units = 80;
    std::vector<int> checkpoints = {
        50'000,
        100'000,
        300'000,
        800'000,
        1'000'000,
        1'500'000,
        2'000'000,
        3'000'000,
        4'000'000,
        5'000'000,
        6'000'000,
        7'000'000,
        8'000'000,
        9'000'000,
        10'000'000,
    };
    int checkpoint = 0;
    int print_frequency = 1'000;

    // Weight filenames
    std::string start_filename = "weights/" + std::to_string(start) + "_games.csv";
    std::string end_filename = "weights/" + std::to_string(end) + "_games.csv";

    // Model
    std::shared_ptr<Model> model = std::make_shared<Model>(hidden_units);

    // Load model
    if (start) {
        model->load(start_filename);
        std::cout << "Loaded weights from file: " << start_filename << std::endl;
    }

    // Game
    Game game(
        std::make_shared<Trainer>("WHITE", model), 
        std::make_shared<Trainer>("BLACK", model)
    );

    // Play games
    for (int i = start + 1; i <= end; i++) {
        if (i % print_frequency == 0) {
            std::cout << "Game nr. " << i << std::endl;
        }
        game.play();

        if (i % checkpoints[checkpoint] == 0) {
            checkpoint++;
            std::string checkpoint = "weights/" + std::to_string(i) + "_games.csv";
            model->save(checkpoint);
            std::cout << "Saved weights in file: " << checkpoint << std::endl;
        }
    }

    // Save model
    model->save(end_filename);

    std::cout << "Saved weights in file: " << end_filename << std::endl;

    return 0;
}
