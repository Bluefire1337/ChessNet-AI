from chess_ai import ChessNet, board_to_tensor, move_to_index



import pygame
import chess
import torch
import os

import sys





# --- PYGAME PARAMETERS ---
WIDTH= 512
HEIGHT = 512
SQ_SIZE = WIDTH // 8
IMAGES = {}


def load_images(folder_images = "chess_assets"):
    """Load images from a folder with images named "color_piecetype" ex: "white_pawn" """
    pieces = ["white_pawn", "white_rook", "white_knight", "white_bishop", "white_queen", "white_king", "black_pawn", "black_rook", "black_knight", "black_bishop", "black_king", "black_queen"]
    for piece in pieces:
        image_path = os.path.join(folder_images, f"{piece}.png")
        IMAGES[piece] = pygame.transform.scale(pygame.image.load(image_path), (SQ_SIZE, SQ_SIZE))


def draw_board(screen):
    # You cannot do colors = [color = pygame.Color("white"), pygame.Color("gray")]! (in general with any class not just pygame, in newer versions there is the := operator which allows you to do this) 
    # A list [ ] is simply a container meant to hold Things (Values/Expressions). If you write [color = pygame.Color("white")], Python gets incredibly confused. It says: "You told me to store a Thing in this list, but you just handed me an Action/Instruction! I don't know how to store an Instruction inside a list!" and it immediately throws a SyntaxError
    colors = [pygame.Color("white"), pygame.Color("gray")]
    for row in range(8):
        for col in range(8):
            color = colors[(col + row) % 2 == 0]
            # pygame.Rect() has no colour (it is an invisible object, just a container for coordinates)
            # In computer graphics (and Pygame), coordinates are always given as (X, Y).
            pygame.draw.rect(screen, color, pygame.Rect((col * SQ_SIZE, row * SQ_SIZE), (SQ_SIZE, SQ_SIZE)))

def draw_pieces(screen, board, flip_board = False):
    """Reads the chess board and draws the pieces on it"""
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # python-chess squares go from bottom-left to top-right. We must flip the Y axis for Pygame.
            if flip_board:
                col = 7 - chess.square_file(square)
                row = chess.square_rank(square)
            else:
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)

            # Map the piece from the chess.board to the image and draw it
            # chess.WHITE is of type bool, same with piece.color;
            color = "white" if piece.color == chess.WHITE else "black"
            # piece.piece_type is an int. chess.piece_name returns the string associated with that int; in the oficial documentation https://python-chess.readthedocs.io/en/latest/_modules/chess.html#piece_name there are 2 lists: 1 with ints and 1 with strings
            piece_type = chess.piece_name(piece.piece_type)

            img_name = f"{color}_{piece_type}"
            # In computer graphics (and Pygame), coordinates are always given as (X, Y).
            screen.blit(IMAGES[img_name], pygame.Rect((col * SQ_SIZE, row * SQ_SIZE), (SQ_SIZE, SQ_SIZE))) # The second argument is the coordinates where the piece is displayed (the top left corner of that Rect)
    
def choose_promotion(screen):
    # 1. Draw a white popup box in the center of the screen
    # Assuming your WIDTH and HEIGHT are around 512, this centers it
    popup_rect = pygame.Rect((100,200), (312, 112))
    pygame.draw.rect(screen, pygame.Color("white"), popup_rect)
    pygame.draw.rect(screen, pygame.Color("black"),  popup_rect, 4) # A thick black border
    # 2. Draw the text instructions 
    font = pygame.font.SysFont(("Arial"), 22, True) # True is bold; loads the font
    text = font.render("Promote! Press Q, R, B or N", True, pygame.Color("black")) # returns a surface, True is antialiasing; color is text color
    screen.blit(text, (115, 240)) # The second argument is the coordinates where the text is displayed on the screen!
    pygame.display.flip() # Force the screen to update right now!
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q: return chess.QUEEN
                if event.key == pygame.K_r: return chess.ROOK
                if event.key == pygame.K_b: return chess.BISHOP
                if event.key == pygame.K_n: return chess.KNIGHT


# --- AI Logic ---
def get_ai_move(board, model, device):
    """Takes the board, asks the AI for its policy, and picks the best legal move."""
    # 1. Convert board to 3D Tensor and add a batch dimension of 1 because board_to_tensor() doesn't add the batch dimension
    # float() because the parameters of the model are float numbers
    # to(device) (cuda) because the ai lives on the gpu and does the math there so you need to give the board to the gpu 
    tensor_board = board_to_tensor(board).unsqueeze(0).float().to(device)

    # 2. Feed it to the brain (No gradients needed for playing!)
    # _ is used when you don't care about a second, third etd output of a function so python throws it into the trash
    with torch.no_grad():
        policy_pred, _ = model(tensor_board)

    # Convert raw output to probabilities; dim 1 = the output classes (neurons, 4672 of them)
    # squeeze() removes the batch dimension (we only want the numbers that the model predicts)
    # cpu() returns the output to the motherboard RAM
    # numpy() converts the pytorch tensor to numpy tensor (which can only exist in the motherboard RAM); takes less space than pytorch tensor and we don't need its additional features anymore
    policy = torch.softmax(policy_pred, dim = 1).squeeze().cpu().numpy()

    # 3. The Legal Move Filter
    best_move = None
    max_prob = -float('inf')

    # Instead of decoding 4672 moves, we just check the probabilities of the ~30 legal moves!
    for move in board.legal_moves:
        move_idx = move_to_index(move)
        prob = policy[move_idx]
        if prob > max_prob:
            max_prob = prob
            best_move = move
    print(f"ChessNet plays {best_move} with {(max_prob * 100):.2f}% confidence.")
    return best_move

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("ChessNet")
    font = pygame.font.SysFont(("Arial"), 22, True) # True is bold; loads the font

    # 1. Load the AI
    # !!!!!!!!!!! Don't forget to change if the model changes !!!!!!!!!!!!!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet().to(device)

    checkpoint_path = os.path.join("checkpoints_batch", "chess_checkpoint_batch_260000.pth")
    print(f"Loading AI Brain from {checkpoint_path}...")
    # When you torch_save(), the location where the model was trained was also saved, and torch.load will also load it. We use map_location = device to change the map location: aka to ignore the location where the model was previously trained.
    # This is a safety feature: What if someone who wants to play against the ai doesn't have cuda? (or whatever device/location the model had when it was trained)
    checkpoint = torch.load(checkpoint_path, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Sets the model to "Evaluation" mode (turns off training features ex: Dropout)
    print("AI is awake and ready!")

    load_images()
    board = chess.Board()

    # --- ADD THESE TWO LINES ---
    HUMAN_COLOR = chess.BLACK  # Change this to chess.BLACK or chess.WHITE to play as either color!
    AI_COLOR = not HUMAN_COLOR # Automatically assigns the opposite color to the AI
    FLIP_BOARD = (HUMAN_COLOR == chess.BLACK) # Flip the board when you are black
    game_over_printed = False

    sq_selected = () # keeps track of the last click (row, col)
    player_clicks = [] # keeps track of player clicks [(row, col), (row, col), etc]

    running = True
    while running:
        # While your game is looping, your Operating System (Windows/Mac) is constantly monitoring your hardware. Every time you wiggle the mouse, press a key, or click the red "X" to close the window, the OS generates an Event. Because your game is busy drawing the board, it can't handle these events instantly. So, Pygame acts as a mailbox. It catches all those hardware events and stuffs them into a waiting line called the Event Queue.
        # When your code hits pygame.event.get(), it does two critical things:
        # It reaches into the mailbox and grabs a list of every single event that happened since the last frame.
        # It completely empties the mailbox so it is ready to catch new events for the next frame.
        # Pygame is an SDL (Simple DirectMedia Layer) wrapper. Look more into that.
        # Basically: events are the translated, software versions of hardware interrupts. pygame.event.get() is a stack of all of them since the last frame. for event in pygame.event.get(): checks them one by one
        # Look more into Hardware Interrupts and Polling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # HUMAN TURN (Assuming Human is White)
            elif event.type == pygame.MOUSEBUTTONDOWN and board.turn == HUMAN_COLOR:
                location = pygame.mouse.get_pos()
                col = location[0] // SQ_SIZE
                row = location[1] // SQ_SIZE

                if sq_selected == (row, col): # User clicked the same square twice
                    sq_selected = ()
                    player_clicks = []
                else:
                    sq_selected = (row, col)
                    player_clicks.append(sq_selected) 
                
                if len(player_clicks) == 2: # After 2nd click
                    # Convert Pygame row/col back to python-chess square index
                    if FLIP_BOARD:
                        from_sq = chess.square(7 - player_clicks[0][1], player_clicks[0][0])
                        to_sq = chess.square(7 - player_clicks[1][1], player_clicks[1][0])
                    else:
                        from_sq = chess.square(player_clicks[0][1], 7 - player_clicks[0][0])
                        to_sq = chess.square(player_clicks[1][1], 7 - player_clicks[1][0])

                    # Dumb vector; doesn't hold info of the picked up piece; chess.board takes care of that
                    move = chess.Move(from_sq, to_sq)
                    if board.piece_at(from_sq) and board.piece_at(from_sq).piece_type == chess.PAWN:
                        if chess.square_rank(to_sq) == 7 or chess.square_rank(to_sq) == 0:
                            # Problem: when you click on pawn and on a bottom square (as white) you stil get the popup (even though you still can't promote)
                            # Technically moving a pawn to row 0 or 7 without promoting it is an illegal move, so you can't check for that
                            # Solution, we force it to promote and ask that if it is an illegal move
                            # PAUSE THE GAME AND ASK THE USER!
                            scout_move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                            if scout_move in board.legal_moves:
                                promoted_piece = choose_promotion(screen)
                                # Build the dumb with the user's choice
                                move = chess.Move(from_sq, to_sq, promotion = promoted_piece)
                    if move in board.legal_moves:
                        board.push(move)
                        sq_selected = ()
                        player_clicks = []
                    else:
                        player_clicks = [sq_selected]

        # AI TURN
        if board.turn == AI_COLOR and not board.is_game_over():
            # Force Pygame to draw the human's move before the AI freezes the screen to think
            draw_board(screen)
            draw_pieces(screen, board, FLIP_BOARD)
            pygame.display.flip()

            ai_move = get_ai_move(board, model, device)
            board.push(ai_move)
            
        # --- CHECK FOR GAME OVER ---
        if board.is_game_over() and not game_over_printed:
            print("\n" + "="*30)
            print("GAME OVER!")
            
            # board.outcome() grabs a rich object with the winner and the exact reason!
            outcome = board.outcome()
            
            if outcome.winner == chess.WHITE:
                print("Result: WHITE WINS!")
            elif outcome.winner == chess.BLACK:
                print("Result: BLACK WINS!")
            else:
                print("Result: IT'S A DRAW!")
                
            # outcome.termination is an Enum that tells us if it was Checkmate, Stalemate, etc.
            print(f"Reason: {outcome.termination.name}")
            print("="*30 + "\n")
            # Flip our safety flag so this never prints again!
            game_over_printed = True
        if board.is_game_over():
            text = font.render("Game over!", True, pygame.Color("black"))
            screen.blit(text, (115, 240))
        draw_board(screen)
        draw_pieces(screen, board, FLIP_BOARD)
        pygame.display.flip()
        pygame.display.flip()   
    pygame.quit()
    sys.exit()

"""
Imagine that next week, you decide you want to build a completely new file called ai_tournament.py to make two different AI models play against each other automatically. You realize you already wrote a beautiful draw_board() function in play_ai.py, and you just want to reuse it without rewriting the code.

So, at the top of your new tournament file, you write:
import play_ai

If you didn't have the if __name__ == "__main__": block, and just left your main() function call sitting naked at the bottom of the file, here is what would happen:
The second Python reads import play_ai, it runs through that entire file from top to bottom. It hits main(), and instantly launches your Pygame window. Your tournament script is completely hijacked, and the game suddenly starts waiting for human mouse clicks!
When Python looks at a file, it secretly assigns a name tag to it using a hidden background variable called __name__.

If you run the file directly: Python gives it the VIP name tag: "__main__".

If you import the file into another script: Python gives it a guest name tag based on the file name (in this case, "play_ai")
"""

if __name__ == "__main__":
    main()
        
            



        
