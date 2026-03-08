# ♟️ ChessNet-AI

A custom-built Deep Learning Chess AI and Graphical User Interface. 

This project features a complete chess engine built from scratch in Python. It uses a **PyTorch Neural Network** to evaluate board states and predict optimal moves, wrapped inside a fully playable **Pygame** interface.

## ✨ Features
* **Playable GUI:** A fully functional, interactive chessboard built with Pygame.
* **FIDE Rule Enforcement:** Strict move validation, pawn promotion logic, and automatic draw detection (50-move rule, threefold repetition, stalemates) handled by `python-chess`.
* **Deep Learning Brain:** A custom neural network trained to predict Grandmaster-level moves based on board state patterns.
* **Scout-Move Logic:** Intelligent UI that verifies legal promotions before prompting the user.

## 🚀 Installation & Setup

To play against the AI on your own machine, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_USERNAME/ChessNet-AI.git](https://github.com/YOUR_USERNAME/ChessNet-AI.git)
cd ChessNet-AI
