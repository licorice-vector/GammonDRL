#include "./game/Game.h"
#include "./player/Player.h"
#include "./player/Human.h"
#include "./player/Trainer.h"

using namespace Backgammon;

int main() {

    int start = 300'000;
    int end = 2'000'000;
    int checkpoint = 200'000;
    
    // Weight filenames
    std::string start_filename = "weights/" + std::to_string(start) + "_games.csv";
    std::string end_filename = "weights/" + std::to_string(end) + "_games.csv";

    // Model
    std::shared_ptr<Model> model = std::make_shared<Model>();

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
        std::cout << "Game nr. " << i << std::endl;
        game.play();

        if (i % checkpoint == 0) {
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
