# ChessNet-AI

A custom-built Deep Learning Chess AI and Graphical User Interface. 

This project features a complete chess engine built from scratch in Python. It uses a **PyTorch Neural Network** to evaluate board states and predict optimal moves, wrapped inside a fully playable **Pygame** interface.

## Features
* **Playable GUI:** A fully functional, interactive chessboard built with Pygame.
* **FIDE Rule Enforcement:** Strict move validation, pawn promotion logic, and automatic draw detection (50-move rule, fivefold repetition, stalemate) handled by `python-chess`.
* **Deep Learning Brain:** A custom neural network trained to predict master level moves based on board state patterns.
* **Scout-Move Logic:** Intelligent UI that verifies legal promotions before prompting the user.

## Installation & Setup

To play against the AI on your own machine, follow these steps:

**1. Clone the repository**
```
git clone [https://github.com/Bluefire1337/ChessNet-AI.git](https://github.com/Bluefire1337/ChessNet-AI.git)
cd ChessNet-AI
```

**2. Install the required dependencies**
```
pip install -r requirements.txt
```

**3. Launch the Game!**
```
python play_ai.py
```
<img width="381" height="406" alt="image" src="https://github.com/user-attachments/assets/57f2f2ea-437a-46cc-9afe-51e7f4c42c01" />

_Note: To change the color you play, modify the HUMAN_COLOR variable inside play_ai.py (for example: from HUMAN_COLOR = chess.BLACK to HUMAN_COLOR = chess.WHITE)_

