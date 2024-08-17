const WHITE = 0;
const BLACK = 1;
const OUT = 24;
const BAR = 25;
const BOARD_SIZE = 24;

const Outcome = {
    WON_SINGLE_GAME: 0,
    WON_GAMMON: 1,
    WON_BACKGAMMON: 2,
    LOST_SINGLE_GAME: 3,
    LOST_GAMMON: 4,
    LOST_BACKGAMMON: 5
};

class CheckerMove {
    constructor(from, to) {
        this.from = from;
        this.to = to;
    }

    static compare(a, b) {
        if (a.from === b.from) {
            return a.to - b.to;
        } else {
            return a.from - b.from;
        }
    }

    static equals(a, b) {
        return a.from === b.from && a.to === b.to;
    }

    static listEquals(a, b) {
        if (a.length !== b.length) return false;
        return a.every((move, index) => CheckerMove.equals(move, b[index]));
    }
}

class State {
    constructor() {
        this.turn = WHITE;
        this.home = [
            [5, 4, 3, 2, 1, 0],
            [18, 19, 20, 21, 22, 23]
        ];
        this.on = [
            Array(BOARD_SIZE + 2).fill(0),
            Array(BOARD_SIZE + 2).fill(0)
        ];
        this.on[BLACK][0] = 2;
        this.on[WHITE][5] = 5;
        this.on[WHITE][7] = 3;
        this.on[BLACK][11] = 5;
        this.on[WHITE][12] = 5;
        this.on[BLACK][16] = 3;
        this.on[BLACK][18] = 5;
        this.on[WHITE][23] = 2;
        this.made = [];
    }

    computePip(player) {
        let pip = 25 * this.on[player][BAR];
        for (let point = 0; point < BOARD_SIZE; point++) {
            pip += this.on[player][point] * Math.abs(point - (player === WHITE ? -1 : 24));
        }
        return pip;
    }

    race() {
        if (this.on[WHITE][BAR] || this.on[BLACK][BAR]) {
            return false;
        }
        let sum = this.on[WHITE][OUT];
        if (sum === 15) {
            return true;
        }
        for (let point = 0; point < BOARD_SIZE; point++) {
            if (this.on[BLACK][point]) {
                return false;
            }
            sum += this.on[WHITE][point];
            if (sum === 15) {
                return true;
            }
        }
        return false;
    }
    
    makeMove(move) {
        let undo = [];
        for (let checkerMove of move) {
            let {from, to} = checkerMove;
            if (to !== OUT && this.on[Number(!this.turn)][to]) {
                this.on[Number(!this.turn)][BAR]++;
                this.on[Number(!this.turn)][to]--;
                undo.push({turn: Number(!this.turn), checkerMove: new CheckerMove(to, BAR)});
            }
            this.on[this.turn][from]--;
            this.on[this.turn][to]++;
            undo.push({turn: this.turn, checkerMove: new CheckerMove(from, to)});
        }
        this.made.push(undo);
    }
    
    undoMove() {
        let undo = this.made.pop();
        for (let {turn, checkerMove} of undo) {
            let {from, to} = checkerMove;
            this.on[turn][to]--;
            this.on[turn][from]++;
        }
    }
    
    getMoves(deltas) {
        let moves = [];
        let move = [];
        if (deltas[0] === deltas[1]) {
            for (let i = 0; i < 4; i++) {
                move = [];
                this.findMoves(deltas, i, move, moves);
                if (moves.length > 0) break;
            }
        } else {
            move = [];
            this.findMoves(deltas, 0, move, moves);
            [deltas[0], deltas[1]] = [deltas[1], deltas[0]];
            move = [];
            this.findMoves(deltas, 0, move, moves);
            if (moves.length === 0) {
                move = [];
                this.findMoves([Math.max(deltas[0], deltas[1])], 0, move, moves);
                if (moves.length === 0) {
                    move = [];
                    this.findMoves([Math.min(deltas[0], deltas[1])], 0, move, moves);
                }
            }
        }
        moves.forEach(m => m.sort(CheckerMove.compare));
        const uniqueMovesSet = new Set();
        const uniqueMoves = [];
        for (let moveSet of moves) {
            const serialized = JSON.stringify(moveSet.map(move => ({from: move.from, to: move.to})));
            if (!uniqueMovesSet.has(serialized)) {
                uniqueMovesSet.add(serialized);
                uniqueMoves.push(moveSet);
            }
        }
        uniqueMoves.forEach(moveSet => moveSet.sort(CheckerMove.compare));
        return uniqueMoves;
    }

    findMoves(deltas, index, move, moves) {
        if (index === deltas.length) {
            moves.push([...move]);
            return;
        }
        let delta = deltas[index];
        if (this.on[this.turn][BAR]) {
            let to = (this.turn === WHITE ? 24 : -1) + (this.turn === WHITE ? -1 : 1) * delta;
            if (to >= 0 && to < BOARD_SIZE && this.on[Number(!this.turn)][to] <= 1) {
                let checker_move = new CheckerMove(BAR, to);
                this.makeMove([checker_move]);
                move.push(checker_move);
                this.findMoves(deltas, index + 1, move, moves);
                move.pop();
                this.undoMove();
            }
            return;
        }
        let bear_off = this.canBearOff();
        for (let from = 0; from < BOARD_SIZE; from++) {
            if (!this.on[this.turn][from]) continue;
            let to = from + (this.turn === WHITE ? -1 : 1) * delta;
            if (to >= 0 && to < BOARD_SIZE && this.on[Number(!this.turn)][to] <= 1) {
                let checker_move = new CheckerMove(from, to);
                this.makeMove([checker_move]);
                move.push(checker_move);
                this.findMoves(deltas, index + 1, move, moves);
                move.pop();
                this.undoMove();
            }
            if (bear_off && this.canBearOffFromDelta(from, delta)) {
                let checker_move = new CheckerMove(from, OUT);
                this.makeMove([checker_move]);
                move.push(checker_move);
                this.findMoves(deltas, index + 1, move, moves);
                move.pop();
                this.undoMove();
            }
        }
    }

    canBearOff() {
        let cnt = 0;
        for (let point of this.home[this.turn]) {
            cnt += this.on[this.turn][point];
        }
        return cnt + this.on[this.turn][OUT] === 15;
    }

    canBearOffFromDelta(from, delta) {
        let direction = (this.turn === WHITE ? -1 : 1);
        let bear_off_point = (this.turn === WHITE ? -1 : 24) - direction * delta;
        if (from === bear_off_point) {
            return true;
        }
        if ((this.turn === WHITE && from > bear_off_point) || (this.turn === BLACK && from < bear_off_point)) {
            return false;
        }
        for (let point = (this.turn === WHITE ? 5 : 18); point != from; point += direction) {
            if (this.on[this.turn][point]) {
                return false;
            }
        }
        return true;
    }

    outcome(player) {
        if (this.on[player][OUT] === 15) {
            if (this.on[!player][OUT]) {
                return Outcome.WON_SINGLE_GAME;
            }
            if (this.on[!player][BAR]) {
                return Outcome.WON_BACKGAMMON;
            }
            for (let point of this.home[player]) {
                if (this.on[!player][point]) {
                    return Outcome.WON_BACKGAMMON;
                }
            }
            return Outcome.WON_GAMMON;
        }
        if (this.on[player][OUT]) {
            return Outcome.LOST_SINGLE_GAME;
        }
        if (this.on[player][BAR]) {
            return Outcome.LOST_BACKGAMMON;
        }
        for (let point of this.home[!player]) {
            if (this.on[player][point]) {
                return Outcome.LOST_BACKGAMMON;
            }
        }
        return Outcome.LOST_GAMMON;
    }

    drawChecker(point, player, index) {
        let w = width / 14;
        let h = height / 2;
        const d = w / 10 * 9;
        fill(player === WHITE ? 'white' : 'black');
        if (point === OUT) {
            if (player === WHITE) {
                let x = (12 + 1) * w;
                x += w / 2;
                let y = height - (index * d + d / 2);
                if (index > 4) {
                    y = height - (4 * d + d / 2);
                    ellipse(x, y, d);
                    fill('white');
                    textSize(24);
                    stroke(0);
                    textAlign(CENTER, CENTER);
                    text(index - 3, x, y);
                } else {
                    ellipse(x, y, d);
                }
                return;
            }
            let x = (point - 12 + 1) * w;
            x += w / 2;
            let y = index * d + d / 2;
            if (index > 4) {
                y = 4 * d + d / 2;
                ellipse(x, y, d);
                fill('white');
                textSize(24);
                stroke(0);
                textAlign(CENTER, CENTER);
                text(index - 3, x, y);
            } else {
                ellipse(x, y, d);
            }
            return;
        }
        if (point === BAR) {
            if (player === WHITE) {
                let x = 6 * w;
                x += w / 2;
                let y = h - d - index * d + d / 2;
                if (index > 4) {
                    y = h - d - 4 * d + d / 2;
                    ellipse(x, y, d);
                    fill('white');
                    textSize(24);
                    stroke(0);
                    textAlign(CENTER, CENTER);
                    text(index - 3, x, y);
                } else {
                    ellipse(x, y, d);
                }
                return;
            }
            let x = (12 - 6) * w;
            x += w / 2;
            let y = h + (index * d + d / 2);
            if (index > 4) {
                y = h + (4 * d + d / 2);
                ellipse(x, y, d);
                fill('white');
                textSize(24);
                stroke(0);
                textAlign(CENTER, CENTER);
                text(index - 3, x, y);
            } else {
                ellipse(x, y, d);
            }
            return;
        }
        if (point < 6) {
            let x = (12 - point) * w;
            x += w / 2;
            let y = height - (index * d + d / 2);
            if (index > 4) {
                y = height - (4 * d + d / 2);
                ellipse(x, y, d);
                fill('white');
                textSize(24);
                stroke(0);
                textAlign(CENTER, CENTER);
                text(index - 3, x, y);
            } else {
                ellipse(x, y, d);
            }
        } else if (point < 12) {
            let x = (12 - point - 1) * w;
            x += w / 2;
            let y = height - (index * d + d / 2);
            if (index > 4) {
                y = height - (4 * d + d / 2);
                ellipse(x, y, d);
                fill('white');
                textSize(24);
                stroke(0);
                textAlign(CENTER, CENTER);
                text(index - 3, x, y);
            } else {
                ellipse(x, y, d);
            }
        } else if (point < 18) {
            let x = (point - 12) * w;
            x += w / 2;
            let y = index * d + d / 2;
            if (index > 4) {
                y = 4 * d + d / 2;
                ellipse(x, y, d);
                fill('white');
                textSize(24);
                stroke(0);
                textAlign(CENTER, CENTER);
                text(index - 3, x, y);
            } else {
                ellipse(x, y, d);
            }
        } else {
            let x = (point - 12 + 1) * w;
            x += w / 2;
            let y = index * d + d / 2;
            if (index > 4) {
                y = 4 * d + d / 2;
                ellipse(x, y, d);
                fill('white');
                textSize(24);
                stroke(0);
                textAlign(CENTER, CENTER);
                text(index - 3, x, y);
            } else {
                ellipse(x, y, d);
            }
        }
    }

    drawPoint(point) {
        let w = width / 14;
        let h = height / 2;
        fill(point % 2 === 0 ? '#6096B4' : '#BDCDD6');
        if (point < 6) {
            let x = (12 - point) * w;
            let y = h;
            triangle(x, h * 2, x + w, h * 2, x + w / 2, y + w);
        } else if (point < 12) {
            let x = (12 - point - 1) * w;
            let y = h;
            triangle(x, h * 2, x + w, h * 2, x + w / 2, y + w);
        } else if (point < 18) {
            let x = (point - 12) * w;
            let y = 0;
            triangle(x, y, x + w, y, x + w / 2, h - w);
        } else {
            let x = (point - 12 + 1) * w;
            let y = 0;
            triangle(x, y, x + w, y, x + w / 2, h - w);
        }
    }

    draw() {
        const w = width / 14;
        const h = height / 2;
        const r = h / 10;
        for (let i = 0; i < BOARD_SIZE; i++) {
            this.drawPoint(i);
            for (let player = 0; player <= 1; player++) {
                for (let j = 0; j < this.on[player][i]; j++) {
                    this.drawChecker(i, player, j);
                }
            }
        }
        fill('#EEE9DA');
        rect(6 * w, 0, w, h);
        fill('#EEE9DA');
        rect(6 * w, h, w, h);
        fill('#EEE9DA');
        rect(13 * w, 0, w, h);
        fill('#EEE9DA');
        rect(13 * w, h, w, h);
        for (let i of [BAR, OUT]) {
            for (let player = 0; player <= 1; player++) {
                for (let j = 0; j < this.on[player][i]; j++) {
                    this.drawChecker(i, player, j);
                }
            }
        }
        fill('white');
        textSize(24);
        stroke(0);
        textAlign(CENTER, CENTER);
        text(this.turn === WHITE ? 'White\'s turn' : 'Black\'s turn', width / 4, height / 2);
        if (mouseX <= 0 || mouseX >= width || mouseY <= 0 || mouseY >= height) {
            // Outside of screen
        } else if (mouseX < 6 * w && mouseY > h) {
            // point 6 - 11
            let point = floor(12 - mouseX / w);
            fill('white');
            let x = (12 - point - 1) * w;
            let y = h;
            ellipse(x + w / 2, y + w * 0.5, 5);
        } else if (mouseX > 7 * w && mouseX < 13 * w && mouseY > h) {
            // point 0 - 5
            let point = floor(6 - (mouseX - 7 * w) / w);
            fill('white');
            let x = (13 - point - 1) * w;
            let y = h;
            ellipse(x + w / 2, y + w * 0.5, 5);
        } else if (mouseX < 6 * w && mouseY < h) {
            // point 12 - 17
            let point = floor(12 + mouseX / w);
            fill('white');
            let x = (point - 12) * w;
            ellipse(x + w / 2, h - w * 0.5, 5);
        } else if (mouseX > 7 * w && mouseX < 13 * w && mouseY < h) {
            // point 18 - 23
            let point = floor(11 + mouseX / w);
            fill('white');
            let x = (point - 12 + 1) * w;
            ellipse(x + w / 2, h - w * 0.5, 5);
        } else if (mouseX > 6 * w && mouseX < 7 * w) {
            let point = BAR;
            fill('white');
            let x = 6 * w;
            ellipse(x + w / 2, h, 5);
        } else if (mouseX > 13 * w) {
            let point = OUT;
            fill('white');
            let x = (point - 12 + 1) * w;
            ellipse(x + w / 2, h, 5);
        }
    }
}

class Dice {
    uniform(low, high) {
        return Math.floor(Math.random() * (high - low + 1)) + low;
    }

    constructor() {
        this.firstThrow = true;
        this.first = 0;
        this.second = 0;
    }

    roll() {
        if (this.firstThrow) {
            this.firstThrow = false;
            return;
        }
        this.first = this.uniform(1, 6);
        this.second = this.uniform(1, 6);
    }

    getDeltas() {
        if (this.first === this.second) {
            return [this.first, this.second, this.first, this.second];
        }
        return [this.first, this.second];
    }

    draw() {
        if (this.first === 0 || this.second === 0) {
            return;
        }
        textSize(32);
        fill('white');
        stroke(0);
        textAlign(CENTER, CENTER);
        text(this.first + ' ' + this.second, width / 10 * 7, height / 2);
    }
}

function array_from_state(state) {
    let values = [];
    for (let player = 0; player <= 1; player++) {
        for (let point = 0; point < BOARD_SIZE; point++) {
            let n = state.on[player][point];
            if (n === 0) {
                values.push(0.0);
                values.push(0.0);
                values.push(0.0);
                values.push(0.0);
                continue;
            }
            if (n === 1) {
                values.push(1.0);
                values.push(0.0);
                values.push(0.0);
                values.push(0.0);
                continue;
            }
            if (n === 2) {
                values.push(1.0);
                values.push(1.0);
                values.push(0.0);
                values.push(0.0);
                continue;
            }
            values.push(1.0);
            values.push(1.0);
            values.push(1.0);
            values.push((n - 3.0) / 2.0);
        }
        values.push(state.computePip(player) / 375.0);
    }
    for (let player = 0; player <= 1; player++) {
        values.push(state.on[player][BAR] / 2.0);
    }
    for (let player = 0; player <= 1; player++) {
        values.push(state.on[player][OUT] / 15.0);
    }
    if (state.turn === WHITE) {
        values.push(1.0);
        values.push(0.0);
    } else {
        values.push(0.0);
        values.push(1.0);
    }
    values.push(Number(state.race()));
    if (values.length !== INPUT_FEATURES) {
        alert('Error! values.lenght !== ' + INPUT_FEATURES);
    }
    return values;
}

let cnv;
let width;
let height;

let dice;
let state;
let gameOver;

let newGameButton;
let rollDiceButton;
let playMoveButton;
let clearMoveButton;
let currentMoveText;
let availableMovesText;

let currentMove;
let selectedPoint;

let loaded = false;

function setup() {
    width = Math.min(windowWidth, windowHeight) / 1.3;
    height = width;
    cnv = createCanvas(width, height);
    cnv.parent('canvas-container');

    strokeWeight(2);
    
    let buttonContainer = createDiv();
    buttonContainer.style('display', 'flex');
    buttonContainer.style('justify-content', 'center');
    buttonContainer.style('align-items', 'center');
    buttonContainer.style('margin', '10px');

    newGameButton = createButton('New Game');
    newGameButton.style('margin', '10px');
    newGameButton.style('font-size', '18px');
    newGameButton.style('padding', '10px 20px');
    newGameButton.mousePressed(startNewGame);

    rollDiceButton = createButton('Roll Dice');
    rollDiceButton.style('margin', '10px');
    rollDiceButton.style('font-size', '18px');
    rollDiceButton.style('padding', '10px 20px');
    rollDiceButton.mousePressed(rollDice);

    playMoveButton = createButton('Play Move');
    playMoveButton.style('margin', '10px');
    playMoveButton.style('font-size', '18px');
    playMoveButton.style('padding', '10px 20px');
    playMoveButton.mousePressed(playMove);

    clearMoveButton = createButton('Clear Move');
    clearMoveButton.style('margin', '10px');
    clearMoveButton.style('font-size', '18px');
    clearMoveButton.style('padding', '10px 20px');
    clearMoveButton.mousePressed(clearMove);

    currentMoveText = createP('Current Move: ');
    currentMoveText.style('text-align', 'center');
    
    availableMovesText = createP('Available Moves: ');
    availableMovesText.style('text-align', 'center');

    buttonContainer.child(newGameButton);
    buttonContainer.child(rollDiceButton);
    buttonContainer.child(playMoveButton);
    buttonContainer.child(clearMoveButton);

    let textContainer = createDiv();
    textContainer.style('text-align', 'center');
    textContainer.style('margin-top', '20px');
    textContainer.child(currentMoveText);
    textContainer.child(availableMovesText);

    createDiv().child(buttonContainer);
    createDiv().child(textContainer);
    
    startNewGame();
}

function draw() {
    background('#93BFCF');

    if (!loaded) {
        return;
    }
    
    if (gameOver) {
        state.draw();
        rollDiceButton.hide();
        playMoveButton.hide();
        clearMoveButton.hide();
        fill('white');
        textSize(24);
        stroke(0);
        textAlign(CENTER, CENTER);
        text('Game Over', width / 10 * 7, height / 2);
    } else {
        state.draw();
        dice.draw();
        if (state.turn === BLACK) {
            if (!rollDice()) {
                return;
            }
            moves = state.getMoves(dice.getDeltas());
            let index = 0;
            let bestProbability = state.turn === WHITE ? -Infinity : Infinity;
            for (let i = 0; i < moves.length; i++) {
                state.makeMove(moves[i]);
                state.turn = Number(!state.turn);
                let probability = predict(array_from_state(state));
                state.undoMove();
                state.turn = Number(!state.turn);
                if (
                    (state.turn === WHITE && bestProbability < probability) ||
                    (state.turn === BLACK && bestProbability > probability)
                ) {
                    bestProbability = probability;
                    index = i;
                }
            }
            currentMove = moves[index];
            playMove();
        }
    }
}
    
function mousePressed() {
    let w = width / 14;
    let h = height / 2;
    if (mouseX <= 0 || mouseX >= width || mouseY <= 0 || mouseY >= height) {
        // Outside of screen
    } else if (mouseX < 6 * w && mouseY > h) {
        // point 6 - 11
        let point = floor(12 - mouseX / w);
        if (selectedPoint === null) {
            selectedPoint = point;
        } else {
            currentMove.push(new CheckerMove(selectedPoint, point));
            updateCurrentMoveText();
            selectedPoint = null;
        }
    } else if (mouseX > 7 * w && mouseX < 13 * w && mouseY > h) {
        // point 0 - 5
        let point = floor(6 - (mouseX - 7 * w) / w);
        if (selectedPoint === null) {
            selectedPoint = point;
        } else {
            currentMove.push(new CheckerMove(selectedPoint, point));
            updateCurrentMoveText();
            selectedPoint = null;
        }
    } else if (mouseX < 6 * w && mouseY < h) {
        // point 12 - 17
        let point = floor(12 + mouseX / w);
        if (selectedPoint === null) {
            selectedPoint = point;
        } else {
            currentMove.push(new CheckerMove(selectedPoint, point));
            updateCurrentMoveText();
            selectedPoint = null;
        }
    } else if (mouseX > 7 * w && mouseX < 13 * w && mouseY < h) {
        // point 18 - 23
        let point = floor(11 + mouseX / w);
        if (selectedPoint === null) {
            selectedPoint = point;
        } else {
            currentMove.push(new CheckerMove(selectedPoint, point));
            updateCurrentMoveText();
            selectedPoint = null;
        }
    } else if (mouseX > 6 * w && mouseX < 7 * w) {
        let point = BAR;
        if (selectedPoint === null) {
            selectedPoint = point;
        } else {
            currentMove.push(new CheckerMove(selectedPoint, point));
            updateCurrentMoveText();
            selectedPoint = null;
        }
    } else if (mouseX > 13 * w) {
        let point = OUT;
        if (selectedPoint === null) {
            selectedPoint = point;
        } else {
            currentMove.push(new CheckerMove(selectedPoint, point));
            updateCurrentMoveText();
            selectedPoint = null;
        }
    }
}

function rollDice() {
    dice.roll();
    rollDiceButton.hide();
    playMoveButton.show();
    clearMoveButton.show();
    moves = state.getMoves(dice.getDeltas());
    updateAvailableMovesText(moves);
    if (moves.length === 0) {
        let s = (state.turn === WHITE ? 'White' : 'Black');
        s += ' has no moves. Dice roll was ' + dice.first + ' ' + dice.second;
        alert(s);
        state.turn = Number(!state.turn);
        rollDiceButton.show();
        playMoveButton.hide();
        clearMoveButton.hide();
        clearMove();
        return false;
    }
    return true;
}

/*
function sleep(millis) {
    const t = Date.now();
    while(Date.now() - t < millis) {}
}
*/

function playMove() {
    //sleep(500);
    currentMove.sort(CheckerMove.compare);
    let moves = state.getMoves(dice.getDeltas());
    let moveIsValid = moves.some(move => CheckerMove.listEquals(move, currentMove));
    if (moveIsValid) {
        state.makeMove(currentMove);
        if (state.on[state.turn][OUT] === 15) {
            gameOver = true;
        }
        state.turn = Number(!state.turn);
        rollDiceButton.show();
        playMoveButton.hide();
        clearMoveButton.hide();
        clearMove();
    } else {
        alert('Move is invalid. Please try again.');
    }
}

function clearMove() {
    currentMove = [];
    updateCurrentMoveText();
    selectedPoint = null;
}

function startNewGame() {
    dice = new Dice();
    state = new State();
    gameOver = false;
    newGameButton.show();
    rollDiceButton.show();
    playMoveButton.hide();
    clearMoveButton.hide();
    currentMove = [];
    updateCurrentMoveText();
    updateAvailableMovesText([]);
    selectedPoint = null;
    while (dice.first === dice.second) {
        dice.first = dice.uniform(1, 6);
        dice.second = dice.uniform(1, 6);
    }
    dice.firstThrow = true;
    if (dice.first < dice.second) {
        state.turn = WHITE;
    } else {
        state.turn = BLACK;
    }
}

function updateCurrentMoveText() {
    let s = 'Current Move: ';
    for (let checkerMove of currentMove) {
        let {from, to} = checkerMove;
        let a = from + 1;
        let b = to + 1;
        if (from === OUT) {
            a = 'OUT';
        } else if (from === BAR) {
            a = 'BAR';
        }
        if (to === OUT) {
            b = 'OUT';
        } else if (to === BAR) {
            b = 'BAR';
        }
        s += '(' + a + ', ' + b + ') ';
    }
    currentMoveText.html(s);
}

function updateAvailableMovesText(moves) {
    let s = 'Available Moves:<br>';
    for (let move of moves) {
        for (let checkerMove of move) {
            let {from, to} = checkerMove;
            let a = from + 1;
            let b = to + 1;
            if (from === OUT) {
                a = 'OUT';
            } else if (from === BAR) {
                a = 'BAR';
            }
            if (to === OUT) {
                b = 'OUT';
            } else if (to === BAR) {
                b = 'BAR';
            }
            s += '(' + a + ', ' + b + ') ';
        }
        s += '<br>';
    }
    availableMovesText.html(s);
}

let buffer;
let W1, b1, W2, b2;
let HIDDEN_SIZE = 80;
let INPUT_FEATURES = 201;
let OUTPUT = 1;

const fileInput = document.getElementById('csv');
const readFile = () => {
    const reader = new FileReader();
    reader.onload = () => {
        load(reader.result);
    };
    reader.readAsText(fileInput.files[0]);
}
fileInput.addEventListener('change', readFile);

function load(text) {
    const lines = text.split('\n');
    const data = lines.flatMap(line => {
        return line.split(',').map(Number);
    });

    let index = 0;
    
    function readMatrix(rows, cols) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            matrix[i] = [];
            for (let j = 0; j < cols; j++) {
                matrix[i][j] = data[index++];
            }
        }
        return matrix;
    }
    
    function readVector(size) {
        const vector = [];
        for (let i = 0; i < size; i++) {
            vector[i] = data[index++];
        }
        return vector;
    }
    
    index++;
    W1 = readMatrix(HIDDEN_SIZE, INPUT_FEATURES);
    index++;
    b1 = readVector(HIDDEN_SIZE);

    index++;
    W2 = readMatrix(OUTPUT, HIDDEN_SIZE);
    index++;
    b2 = readVector(OUTPUT);

    loaded = true;
}

function relu(value) {
    return Math.max(0, value);
}

function sigmoid(value) {
    if (0.0 < value) {
        return 1.0 / (1.0 + Math.exp(-value));
    } else {
        const exp_value = Math.exp(value);
        return exp_value / (1.0 + exp_value);
    }
}

function predict(x) {
    let layer1 = Array(HIDDEN_SIZE).fill(0.0);
    for (let i = 0; i < HIDDEN_SIZE; i++) {
        for (let j = 0; j < INPUT_FEATURES; j++) {
            layer1[i] += W1[i][j] * x[j];
        }
        layer1[i] += b1[i];
    }
    layer1 = layer1.map(relu);
    let layer2 = Array(OUTPUT).fill(0.0);
    for (let i = 0; i < OUTPUT; i++) {
        for (let j = 0; j < HIDDEN_SIZE; j++) {
            layer2[i] += W2[i][j] * layer1[j];
        }
        layer2[i] += b2[i];
    }
    layer2 = layer2.map(sigmoid);
    return layer2[0];
}
