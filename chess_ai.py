import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import zstandard as zstd
import io

import numpy as np

import glob 

import os
# Helper functions to convert moves to indices; see "move_to_index(move)" function below
def move_is_knight_jump(from_square, to_square):
    from_rank = chess.square_rank(from_square)
    from_file = chess.square_file(from_square)
    to_rank = chess.square_rank(to_square)
    to_file = chess.square_file(to_square)
    rank_diff = abs(to_rank - from_rank)
    file_diff = abs(to_file - from_file)
    return (rank_diff, file_diff) in [(2, 1), (1, 2)]

def knight_jump_index(from_square, to_square):
    from_rank = chess.square_rank(from_square)
    from_file = chess.square_file(from_square)
    to_rank = chess.square_rank(to_square)
    to_file = chess.square_file(to_square)
    rank_diff = to_rank - from_rank
    file_diff = to_file - from_file
    knight_moves = [
        (1, 2), (-1, 2), (1, -2), (-1, -2),
        (2, 1), (-2, 1), (2, -1), (-2, -1)
    ]
    if (rank_diff, file_diff) in knight_moves:
        return knight_moves.index((rank_diff, file_diff))
    else:
        raise ValueError("Not a knight jump")

def sliding_move_index(from_square, to_square):
    from_rank = chess.square_rank(from_square)
    from_file = chess.square_file(from_square)
    to_rank = chess.square_rank(to_square)
    to_file = chess.square_file(to_square)
    rank_diff = to_rank - from_rank
    file_diff = to_file - from_file
    distance = max(abs(rank_diff), abs(file_diff))
    step = (rank_diff // distance, file_diff // distance) if distance != 0 else (0, 0)
    directions = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1)
    ]
    direction_index = directions.index(step) if step in directions else -1
    if direction_index == -1 or distance < 1 or distance > 7:
        raise ValueError("Not a valid sliding move")
    return direction_index * 7 + (distance - 1) #distance can't be 0, it starts from 1, so we subtract 1 to make it 0-indexed

def board_to_tensor(board):
    tensor = torch.zeros(18, 8, 8, dtype=torch.uint8)  # 18 channels: 12 for pieces, 1 for turn, 4 for castling rights, 1 for en passant; each channel is a binary 8x8 matrix; 1 if the piece is present, 0 otherwise; the dtype is uint8 to save disk space (we only need 0 and 1); default is float32
    # Originally there were only 12 channels for 12 pieces. That is NOT enough information because the model would see the same board position for white's turn and black's turn and classify it as the same state and mess up the gradient. It is NOT the same state. Whose turn it is can have a huge impact on the outcome of the game.
    # 1. 12 channels for each type of piece
    piece_to_channel = { 
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = piece_to_channel[(piece.piece_type, piece.color)]
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[channel, rank, file] = 1
    
    # 2. Whose turn is it? (Channel 12)
    # Fill the entire 8x8 channel with 1s if it's White's turn, 0s if Black's
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1

    # 3. Castling rights (Channels 13-16)
    if board.has_kingside_castling_rights(chess.WHITE): tensor[13, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE): tensor[14, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK): tensor[15, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK): tensor[16, :, :] = 1

    # 4. En Passant square (Channel 17)
    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        tensor[17, rank, file] = 1
        
    return tensor
        
board = chess.Board()
tensor_representation = board_to_tensor(board)
#print(tensor_representation)

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(18, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Fully connected layers for policy (best move) and value (game outcome) heads
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)  # Policy head for move probabilities; 2 channels for which piece to pick up and where to move it
        self.policy_output = nn.Linear(2 * 8 * 8, 4672) #https://arxiv.org/pdf/1712.01815 Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (8 × 8 × 73 = 4672 outputs) 
        #Pretty much, to pick up a piece, there are 64 squares to choose from. 
        #It is easy to treat each possible piece as a queen,
        #which has 8 directions and 7 squares in each directions to move to. 
        #Add 8 possible knight moves for the theoretical queen piece, 
        #plus 9 possible ways to underpromote a pawn (queen promotion is already taken into account in the queen movement) 
        #and you get 73 ways a piece can "move". 
        #Treating each piece as a queen, castling is also taken care of.
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)  # Value head for game outcome
        self.value_lin1 = nn.Linear(1 * 8 * 8, 32)
        self.value_lin2 = nn.Linear(32 , 1)  # Single output for game value
        # 2 Linear opperations so that we can have a non-linear activation function in between
    def forward(self, x):
        x = F.relu(self.conv1(x)) #calls the __call__ method of nn.Module, which in turn calls the forward method
        x = F.relu(self.conv2(x))

        # Policy head forward pass
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(-1, 2 * 8 * 8)  # Flatten; x is (batch_size, 64, 8, 8) and we flatten it to (batch_size, 64 * 8 * 8)
        policy = self.policy_output(policy) # Final output for policy head
        # Do NOT apply ReLu here (at the final output). Our loss class applies CrossEntropyLoss (and uses softmax for probabilities) which requires all the logits unchanged (that also means negative logits!), if we want the true predicted probabilities
        # Value head forward pass
        value = F.relu(self.value_conv(x))
        value = value.view(-1, 1 * 8 * 8)
        value = F.relu(self.value_lin1(value))  # Final output for value head
        value = torch.tanh(self.value_lin2(value)) # Normalize value output to [-1, 1]
        return policy, value
    

#the number of indices is 4672, equal to the number of outputs in the policy head
def move_to_index(move):
    from_square=move.from_square
    to_square=move.to_square
    # promotion to queen is treated as a normal sliding move, so no special case for that
    if move.promotion and move.promotion != chess.QUEEN:
        promotion_piece = move.promotion
        from_file=move.from_square % 8
        to_file=move.to_square % 8
        delta_file=to_file - from_file
        if delta_file == 0:
            direction = "forward"
        elif delta_file == -1:
            direction = "capture_left" 
        elif delta_file == 1:
            direction = "capture_right"
        else:
            raise ValueError("Invalid promotion move")
        #we can't have negative indices, so we map the piece and direction to a unique non-negative index; if we had negative indices, it would be possible that 2 moves would map to the same index
        piece_map={
            chess.ROOK:0,
            chess.BISHOP:1,
            chess.KNIGHT:2
        }
        direction_map={
            "forward":0,
            "capture_left":1,
            "capture_right":2
        }

        underpromotion_offset = piece_map[promotion_piece] * 3 + direction_map[direction]
        move_type_offset = underpromotion_offset + 64 #Underpromotions = 64–72 (sliding moves = 0–55, knight moves = 56–63) see below
    elif move_is_knight_jump(from_square, to_square):
        knight_jump_offset = knight_jump_index(from_square, to_square)
        move_type_offset = knight_jump_offset + 56 #Knight moves = 56–63 (sliding moves = 0–55) see below 
    else:
        sliding_move_offset = sliding_move_index(from_square, to_square)
        move_type_offset = sliding_move_offset #Sliding moves = 0–55 (Queen moves)
    index = from_square * 73 + move_type_offset
    return index

#Example usage of move_to_index function        
# moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("g1f3"), chess.Move.from_uci("a7a8n")]
# for m in moves:
#     idx = move_to_index(m)
#     print(m, idx, idx // 73, idx % 73)

# # # # NOT USED  (it was used when I was reading the pgn file in chunks of lines, but now I read it game by game; in the process_large_pgn function below)
# # The chess library has "game" objects that contain functions which return specific parts of the pgn text such as the result of the game, and can transform pgn moves to UCI moves, which is very useful in training the model
# # So, we transform the text pgn file into "chess game" objects which is the parameter of the process_game function used below 
# def process_pgn_game(pgn_lines):
#     pgn_text = "".join(pgn_lines) # Join list of lines into a single string ("" is the character used to join the lines; here we use an empty string, so no extra characters are added between lines; this works because each line already ends with a newline character)
#     pgn_io = io.StringIO(pgn_text) # Create a StringIO object to simulate a file (a file-like object)
#     game = chess.pgn.read_game(pgn_io) # Read the game from the StringIO object; this function only accepts file-like objects
#     if game is None:
#         return []
#     else:
#         return game

def process_game(game, min_rating=2200):
    white_elo = int(game.headers.get("WhiteElo", 0))
    black_elo = int(game.headers.get("BlackElo", 0))
    if white_elo < min_rating or black_elo < min_rating:
        return []  # Skip games where either player has an Elo rating below 2200; There are many bad games in the pgn file, so we filter them out; Also I am afraid of using too much disk space, so I want to limit the number of training samples
    training_samples = []

    #1. Get the game outcome
    result = game.headers["Result"]
    if result == "1-0":
        value = 1  # White wins
    elif result == "0-1":
        value = -1  # Black wins
    elif result == "1/2-1/2":
        value = 0  # Draw
    else:
        return []  # Skip games with unknown results
    
    #2. Iterate through the moves and generate training samples
    board = game.board()
    for move in game.mainline_moves():
        # Convert the current board state to a tensor
        board_tensor = board_to_tensor(board)
        # Convert the move to an index
        move_index = move_to_index(move)
        # Get value from the current player's perspective
        # If it's white's turn, the value is as is. If black's, it's inverted. (This evaluation is similar to AlphaZero's style. Stockfish for example has an absolute evaluation: 1.0 always means that white is winning no matter whose turn it is.)
        current_player_value = value if board.turn == chess.WHITE else -value
        # Store the training sample (board state, move index, game outcome)
        training_samples.append((board_tensor, move_index, current_player_value))
        # Make the move on the board
        board.push(move)
    return training_samples

file_input = 'lichess_db_standard_rated_2025-07.pgn.zst'
file_output = 'lichess_db_standard_rated_2025-07.pgn'



# Saving board states and other things in npy files in CHUNKS so I only have one chunk at a time in RAM (the pgn file is very big and I am scared)
def save_chunk(chunk_id, boards, policies, values):
    boards_np = np.stack(boards, dtype=np.uint8) # The tensor only has 0 and 1 as data; see above at board_to_tensor function
    policies_np = np.array(policies, dtype=np.int16) # The maximum index is 4672 (see the number of output neurons in policy_output)
    values_np = np.array(values, dtype=np.int8) # The values are only -1, 0 and 1

    # Creates a binary file automatically
    np.save(f"chess_numpy_chunks/boards_{chunk_id}.npy", boards_np)
    np.save(f"chess_numpy_chunks/policies_{chunk_id}.npy", policies_np)
    np.save(f"chess_numpy_chunks/values_{chunk_id}.npy", values_np)

    print(f"Saved chunk {chunk_id}: {boards_np.shape[0]} samples")


# Processing each PGN game without loading entire file into RAM (pgn file is very large)
def process_large_pgn(file_input, chunk_size=100000):
  dctx= zstd.ZstdDecompressor()
  # Get the next chunk ID based on existing files
  existing_chunks = glob.glob("chess_numpy_chunks/boards_*.npy")
  chunk_id = len(existing_chunks)  # Start from the next chunk ID
  print("Starting from chunk ID:", chunk_id)
  boards, policies, values = [], [], []
  with open(file_input, 'rb') as compressed_file:
      with dctx.stream_reader(compressed_file) as reader:
          text_stream = io.TextIOWrapper(reader, encoding='utf-8') # Wrap the decompressed byte stream in a text stream, which is a file-like object that chess.pgn.read_game can read from
          while True:
              game=chess.pgn.read_game(text_stream) #this function reads one game at a time from the text stream
              if game is None:
                  break
              samples = process_game(game)
              for board_tensor, move_index, game_value in samples:
                  boards.append(board_tensor.numpy())  # Convert tensor to numpy array for saving
                  policies.append(move_index)
                  values.append(game_value)
              if len(boards) >= chunk_size:
                  save_chunk(chunk_id, boards, policies, values)
                  chunk_id += 1
                  boards, policies, values = [], [], []  # Reset for next chunk; this way we do not keep all samples in RAM
          if boards:  # Save any remaining samples if the last chunk is not full
              save_chunk(chunk_id, boards, policies, values)

# We are making a custom dataset class which we will plug in the DataLoader (to load the data in batches during training and shuffle it and other useful things which we won't implement ourselves) 
# We could theoretically make a for loop that goes through all the chunks and loads them one by one (so we don't fill our RAM) and train the model, but the Dataset and Dataloader classes from Pytorch have many useful features such as shuffling the data, loading data in parallel using multiple workers, etc; which we do not have to implement ourselves
# Moreoever, making a custom Dataset class that inherits from torch.utils.data.Dataset allows us to define how to access each slice/part/chunk by index
class ChessDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # Store the list of all chunk files (it's only the file names, not the actual data; we will load the data in __getitem__ method)
        # Using glob to find all files that match the pattern even if they are not in the working directory
        # Sorting (here lexicographically) to ensure that all the files are in the SAME order (so that boards_0.npy matches policies_0.npy and values_0.npy) because glob returns file names in arbitrary order
        self.board_files = sorted(glob.glob(f"{data_dir}/boards_*.npy"))
        self.policy_files = sorted(glob.glob(f"{data_dir}/policies_*.npy"))
        self.value_files = sorted(glob.glob(f"{data_dir}/values_*.npy"))

        # The transform parameter can be used to apply any data augmentation or preprocessing to the board states (for example, random rotations or flips); yes, this is a function, not a variable
        self.transform = transform

        # Calculating the total number of board states in all chunks
        self.sizes = [np.load(f, mmap_mode="r").shape[0] for f in self.board_files] # mmap_mode="r" allows us to read the shape of the array without loading the entire file into RAM (it only loads the file header which contains information such as the shape and other metadata AND it also loads into RAM whatever you access, in this case shape[0]; "r" is for read only); by default mmap_mode is None, which means the entire file is loaded into RAM
        self.cumulative_sizes = np.cumsum(self.sizes) # Cumulative sum to get the total number of samples up to each chunk; for example, if sizes = [100, 200, 150], then cumulative_sizes = [100, 300, 450]; this will help us find which chunk a particular index belongs to in the __getitem__ method (here an index represents a particular board position in the dataset)
    
    # !!!!! for the dataset to work with the pytorch dataloader, it has to have a __len__ method and __getitem__ method !!!!!

    def __len__(self):
        return self.cumulative_sizes[-1]  # Total number of samples across all chunks ("-1" means the last element of cumulative_sizes)
        
    def __getitem__(self, idx):
        chunk_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') # Find the chunk that contains the idx-th sample; side='right' means if idx is exactly equal to a cumulative size, we take the next chunk (this is important because indices are 0-based and cumulative sizes are 1-based (the first board is added as 1 towards the cumulative sum, not 0))
        # numpy.searchsorted returns the index of the first element in cumulative_sizes that is greater than idx; for example, if cumulative_sizes = [100, 300, 450] and idx = 150, then chunk_idx = 1 (the second chunk); if idx = 100, then chunk_idx = 1 (the second chunk) because we are using side='right'
        if chunk_idx == 0:
            sample_idx = idx # If it's the first chunk, the index of the board positionwithin chunk it belongs to is the same as idx
        else:
            sample_idx = idx - self.cumulative_sizes[chunk_idx - 1] # Find the index within the chunk (if it's not the first chunk)
        
        # Load the specific chunk files (only the chunk that contains the idx-th sample)
        boards = np.load(self.board_files[chunk_idx], mmap_mode="r") # Load the board states for the chunk
        policies = np.load(self.policy_files[chunk_idx], mmap_mode="r") # Load the move indices for the chunk
        values = np.load(self.value_files[chunk_idx], mmap_mode="r") # Load the game outcomes for the chunk

        board=boards[sample_idx] / 1.0 # Convert to float tensor (the board is originally a uint8 tensor with values 0 and 1; we divide by 1.0 to convert it to float; this is important because the model expects float inputs)
        policy=policies[sample_idx]
        value=values[sample_idx]

        if self.transform:
            board = self.transform(board) # Apply any data augmentation or preprocessing to the board state
        # Convert to torch tensors from numpy tensors
        # The policy is a single integer representing the index of the best move, so we convert it to a long tensor (int64) because this is the expected type for classification targets in PyTorch
        # The value is a single uint8 (-1, 0, 1; the game result) representing the game outcome, so we convert it to a float tensor (float32) for the loss function
        # torch.from_numpy creates a tensor that shares memory with the numpy array, so it's efficient; the boards take a lot of space, so we don't want to copy them unnecessarily; their dtype is already float32 because we divided by 1.0 above
        # The policy and value are small, so we can afford to copy them into new tensors (we could use torch.from_numpy here as well, but it's not necessary and the dtype is more important, so we use torch.tensor to specify the dtype, otherwise we would have to convert the arrays afterwards and makes it messy)
        return (torch.from_numpy(board), torch.tensor(policy, dtype=torch.long), torch.tensor(value, dtype=torch.float32))
    

# print(os.getcwd()) # Print the current working directory to ensure we are in the right place

class ChessAiLoss(nn.Module):
    def __init__(self, policy_weight = 1.0, value_weight = 1.0):
        super().__init__()
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        self.policy_weight = policy_weight
        self.value_weight = value_weight

    def forward(self, policy_pred, value_pred, policy_target, value_target):
        policy_loss = self.policy_loss_fn(policy_pred, policy_target) # policy_pred is of shape (batch_size, 4672) and policy_target is of shape (batch_size); the target contains the indices of the correct class for each input in the batch; # This returns a float number
        value_loss = self.value_loss_fn(value_pred.squeeze(), value_target.float()) # value_pred is of shape (batch_size, 1) and value_target is of shape (batch_size); we use squeeze to remove the extra dimension; the target contains the actual game outcome values; float() to ensure the target is float32
        total_loss = self.policy_weight * policy_loss + self.value_weight * value_loss  # Combine both losses; we can also weight them differently if needed
        return total_loss, policy_loss, value_loss
    
def policy_accuracy(policy_pred, policy_target):
    predicted_indices = torch.max(policy_pred, dim=1) # returns a tuple (values, indices); we only need the indices
    correct = (predicted_indices[1] == policy_target).sum().item() # Count of correct predictions; predicted_indices[1] contains the indices of the predicted classes; predicted_indices[1] == policy_target performs element-wise comparison (numpy tensors) and returns a boolean tensor; sum() counts the number of True values (True = 1, False = 0 and adds each of those); item() converts the single-value tensor to a Python number for easier printing
    total = policy_target.size(0) # Total number of samples in the batch; size(0) returns the size of the first dimension (batch size); size() returns something like torch.Size([batch_size]) and size(0) returns the batch_size
    return correct / total

def value_accuracy(value_pred, value_target):
    # value_pred is a tensor computed by the model, so it has gradients; we don't want that when computing accuracy, so we detach it from the computation graph, that is what detach() is for. IF we move it to the cpu first, the gradients calculate are still there, and we won't be able to convert it to numpy array;
    # This goes hand in hand with cpu(), because the tensor might be on the GPU (if we use as device cuda), and we need to move it to the CPU to convert it to numpy array (numpy only supports cpu); 
    # Finally, we use sign() to get the sign of the value prediction (-1, 0, 1), which represents the predicted game outcome from the current player's perspective
    # long() converts the boolean tensor to long tensor (int64);
    pred_sign = value_pred.detach().cpu().sign().long()
    # we do the same for the target values; value_target is already a cpu tensor (because it's not computed by the model), but we still need to detach it (even if it doesn't have gradients, it's a good practice to detach it to avoid any potential issues in the future)
    target_sign = value_target.detach().cpu().sign().long()
    # pred_sign == target_sign performs element-wise comparison (pytorch tensors) and returns a boolean tensor; sum() counts the number of True values (True = 1, False = 0 and adds each of those); item() converts the single-value tensor to a Python number for easier printing
    correct = (pred_sign == target_sign).sum().item()
    total = value_target.size(0) # Total number of samples in the batch
    return correct / total


# process_large_pgn(file_input)

#dataset = ChessDataset("chess_numpy_chunks") # Assuming the npy files are in the "chess_numpy_chunks" directory
# we have made this custom dataset so that it can be wrapped inside DataLoader
#dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True) # Using 4 workers to load data in parallel; this speeds up data loading significantly; 4*num_GPU is a good starting point; shuffle=True to shuffle the data at every epoch; pin_memory=True to speed up transfer of data to GPU if using GPU
# pin_memory= True means the data loader will copy the tensors directly into pinned memory; How transfering data from CPU to GPU works: Data is first copied from regular CPU memory to pinned memory, which is a special type of memory that allows for faster transfer to the GPU. Then, the data is transferred from pinned memory to GPU memory. 
# By using pin_memory=True, we skip the first step of copying data from regular CPU memory to pinned memory, which can save time and improve performance when training models on a GPU.

def decode_move(board, move_index):
    """
    Converts a predicted move_index back into a chess.Move object
    by filtering legal moves on the current board.
    """
    # docstring to display what the function does when the cursor is on it (and what help(function) will return)
    # 1. Deduce the from_square
                
    from_square = move_index // 73
    
    # 2. Iterate over all legal moves for the current board
    # This is safer than writing a reverse mathematical function
    for move in board.legal_moves:
        if move.from_square == from_square:
            # Re-encode this legal move to see if it matches the prediction
            try:
                if move_to_index(move) == move_index:
                    return move
            except ValueError:
                continue
    # If no legal move matches (the AI attempted an illegal move),
    # we can fallback to returning None or selecting a random legal move
    # we don't need the from_square, but this is done for speed so as not to use move_to_index on every single legal move on the board (there can be 20 legal moves at one time)
    return None

def train_model(device, model, dataloader, optimizer, loss_fn, total_epochs, save_dir_batch = "checkpoints_batch", start_batch = 1, save_dir_epoch = "checkpoints_epoch", start_epoch = 0 ):

    #### once used to be inside the function body, now they are outside to make the code more cohesive
    #### 1. Setup device and hyperparameters (not goot to do model = ChessNet().to(device) here: look inside the shield to see why) (technically you can but see the explanation below)
    #device = torch.device("cuda" if torch.cuda.is_available else "cpu") # Context manager that changes the selected device

    # Ensure our save folder exists; this function creates the folder if it doesn't exist already. If it already exists, the program runs normally (exist_ok = True prevents it from crashing in this case)
    os.makedirs(save_dir_batch, exist_ok = True)
    os.makedirs(save_dir_epoch, exist_ok = True)
    print(f"Starting training on {device} from epoch {start_epoch} till epoch {total_epochs}")


    #### once used to be inside the function body, now they are outside to make the code more cohesive
    #BATCH_SIZE = 256 # Decrease if you get Out Of Memory errors
    #NUM_WORKERS = 4
    #LEARNING_RATE = 1e-3

    #### 2. Prepare data
    #dataset = ChessDataset("chess_numpy_chunks")
    #dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS, pin_memory = True)
    #NUM_EPOCHS = 10
    #### 3. Initialize model, optimiser and loss function
    # model = ChessNet().to(device)
    # Stochastic Gradient Descent (SGD) is slow because it only looks at the current gradient. (a small gradient or a zigzag pattern make the training process inefficient)
    # Adam: Instead of just taking isolated steps, Adam remembers the previous steps it took. If the algorithm notices that it has been walking downhill in the exact same direction for the last 100 steps, it says: "I'm probably on a long, clear slope." It builds up momentum and starts taking larger and larger strides.
    # Why it helps: If the landscape briefly levels out or has a tiny bump (a local minimum), Adam has enough built-up speed to blast right through it, preventing the network from getting stuck.
    # In ChessNet, there are over a million different weights (parameters). Some of these weights control obvious things (like "Queen gets captured = bad"), while others control incredibly subtle things (like "White has a slight pawn structure advantage on the queenside").
    # Standard SGD forces every single weight to learn at the exact same speed (your LEARNING_RATE).
    # Adam acts as a smart suspension system. It calculates a custom, individual learning rate for every single weight
    #optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    #criterion = ChessAiLoss()


    for epoch in range(start_epoch, total_epochs + 1):
        #1. Initializing variables
        model.train() # sets the model in "training mode" (activates some features useful for training like Dropout); when you first make the model, pytorch sets it automatically in training mode so it isn't necessary per se right now but it's good practice especially when you consider model.eval() later
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0

        #2. The actual training starts for each batch 
        # One run through this loop is a batch

        for batch_id, (boards, policies, values) in enumerate(dataloader, start = 1):

            # --- THE FAST FORWARD --- (for starting from the last checkpoint where we left off)
            # If we are in the resuming epoch, and we haven't reached our batch yet, skip!
            # !!!!!!!!!!!!!!!! yeah ok, the math the gpu does (loss, optimiser updating parameters, etc) takes much less time than whatever dataloader is doing (opening npy files, bringing them to cpu, etc) so this "fast forwarding" is basically pointless
            if epoch == start_epoch and batch_id <= start_batch:
               
                if batch_id % 500 == 0:
                    print(f"Fast-forwarding... skipped batch {batch_id} / {start_batch}")
                continue

            # # Convert back from uint AND move to GPU; My numpy chunks are saved as int types to save memory but my neural net wants them different (see ChessNet)
            # boards = boards.float().to(device)
            # The model's weights are 32-bit floats. So, the input board must also be converted to a 32-bit float right before it goes into the model(boards) function.
            boards = boards.float().to(device)
            # This is critical! Because your policies are actual index numbers (like move 4672), you are using CrossEntropyLoss. In PyTorch, CrossEntropyLoss absolutely demands that class indices be 64-bit integers (which PyTorch calls a LongTensor). If you try to make this a float, your loss function will instantly crash! You must use .long()
            policies = policies.long().to(device)
            # Our value head predicts who is winning. Because this is calculated using Mean Squared Error (MSELoss) comparing the prediction to -1, 0, or 1, both the prediction and the target must be 32-bit floats. You must use .float()
            values = values.float().to(device)

            # We make the gradient zero. (necessary because pytorch adds to the gradient each batch and we don't want it to explode)
            optimizer.zero_grad()

            policy_pred, value_pred = model(boards)

            loss, policy_loss, value_loss = loss_fn(policy_pred, value_pred, policies, values)
            loss.backward() # We calculate the gradient of the loss function (the sum of the policy loss and value loss)

            # Gradient clipping (prevents the gradient from exploding, another measure to make sure they are not too big)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

            # We update the parameters
            optimizer.step()

            # We update the loss variables initialized earlier (key point: we add the NUMBERS, not tensors, which loss_fn returns, so we use item())
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()


            if (batch_id % 100 == 0):
                print(f"Epoch {epoch} | Batch {batch_id} | Loss: {loss.item():.4f} | Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f} | Policy Acc: {policy_accuracy(policy_pred, policies):.2f}")
            if (batch_id % 20000 == 0):
            
                # 3. THE TRUE CHECKPOINT SYSTEM # used to be only once every epoch but the program crashed before finishing an epoch :)
                checkpoint_batch_path = os.path.join(save_dir_batch, f"chess_checkpoint_batch_{batch_id}.pth")

                # Bundle the brain, the momentum, the epoch, and the score into a master dictionary; no avg_loss here
                checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }
                # Save it safely to the hard drive
                # While the dictionary itself is standard Python, the contents (model.state_dict()) are highly specialized PyTorch Tensors. That is why we use torch.save().
                # Here is the golden rule: model.parameters() is for the Optimizer, and model.state_dict() is for the Hard Drive.
                torch.save(checkpoint, checkpoint_batch_path)
                print(f"--> Saved True Checkpoint to {checkpoint_batch_path}\n")
        # End of epoch stats
        avg_loss = total_loss / len(dataloader)
        avg_policy_loss = total_policy_loss / len(dataloader)
        avg_value_loss = total_value_loss / len(dataloader)
            
        # 3. THE TRUE CHECKPOINT SYSTEM (once every epoch)
        checkpoint_epoch_path = os.path.join(save_dir_epoch, f"chess_checkpoint_epoch_{epoch}.pth")

        # Bundle the brain, the momentum, the epoch, and the score into a master dictionary; no avg_loss here
        checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'policy_loss': avg_policy_loss,
        'value_loss': avg_value_loss
        }
        # Save it safely to the hard drive
        # While the dictionary itself is standard Python, the contents (model.state_dict()) are highly specialized PyTorch Tensors. That is why we use torch.save().
        torch.save(checkpoint, checkpoint_epoch_path)
        print(f"--> Saved True Checkpoint to {checkpoint_epoch_path}\n")
        print(f"--> Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f} Avg_policy_loss: {avg_policy_loss:.4f} Avg_value_loss:{avg_value_loss:.4f}")

    
    print("Training Complete!")




save_dir_batch  = "checkpoints_batch"
save_dir_epoch = "checkpoints_epoch"
start_batch = 0
start_epoch = 1

# To study another time
""" 
    The Problem: The Infinite Clone Army
When you set num_workers = 4 in your DataLoader, you are telling PyTorch to create 4 separate background processes to load your chess data.
Here is where the Operating System wars strike again:
On Linux/Mac: The OS uses a command called fork, which instantly clones the memory exactly as it is. It's safe and quiet.
On Windows: Windows doesn't have fork. Instead, it uses spawn. To create a new worker, Windows literally opens a brand new, blank Python window and re-runs your entire script from top to bottom to load all the functions into its memory.
Because your train_model() command is just sitting out in the open, Worker #1 wakes up, reads your script, and says: "Oh, I'm supposed to start training!" So Worker #1 creates a DataLoader, which creates 4 more workers, which read the script, which create 4 more workers...
Your computer realizes it is building an infinite army of clones and pulls the emergency brake, throwing that RuntimeError.

The Fix: The Shield
To fix this, you have to put all of your "action" code behind a protective shield so that the background workers can't accidentally run it.

In Python, that shield is the if __name__ == '__main__': block.
When Windows spawns a background worker, its "name" is technically not '__main__', so the worker will safely skip over everything inside that block!
"""



# --- THE SHIELD ---
if __name__ == '__main__':

    #1. Setup device and hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available else "cpu") # Context manager that changes the selected device
    BATCH_SIZE = 128 # Decrease if you get Out Of Memory errors; after like 25000 batches the program crashed with 256 and 4 workers so I will lower these values
    NUM_WORKERS = 2
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10

    #2. Prepare data
    dataset = ChessDataset("chess_numpy_chunks")
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS, pin_memory = True)
    
    #3. Initialize model, optimiser and loss function
    ########## The model parameters need to be moved to the gpu before you use the optimizer! The optimizer stores the addreses of those and if you move it after you have written  optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE), (for ex: in the body function) the optimizer will update the CPU addresses while the brain of the net is actually on the gpu! The net (which is on the gpu) won't learn anything because the parameters aren't updated.
    model = ChessNet().to(device)

    # Stochastic Gradient Descent (SGD) is slow because it only looks at the current gradient. (a small gradient or a zigzag pattern make the training process inefficient)
    # Adam: Instead of just taking isolated steps, Adam remembers the previous steps it took. If the algorithm notices that it has been walking downhill in the exact same direction for the last 100 steps, it says: "I'm probably on a long, clear slope." It builds up momentum and starts taking larger and larger strides.
    # Why it helps: If the landscape briefly levels out or has a tiny bump (a local minimum), Adam has enough built-up speed to blast right through it, preventing the network from getting stuck.
    # In ChessNet, there are over a million different weights (parameters). Some of these weights control obvious things (like "Queen gets captured = bad"), while others control incredibly subtle things (like "White has a slight pawn structure advantage on the queenside").
    # Standard SGD forces every single weight to learn at the exact same speed (your LEARNING_RATE).
    # Adam acts as a smart suspension system. It calculates a custom, individual learning rate for every single weight
    # Here is the golden rule: model.parameters() is for the Optimizer, and model.state_dict() is for the Hard Drive.
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    criterion = ChessAiLoss()

    # --- THE RESUME CODE --- (comment this if you start from scratch)
    resume_checkpoint_path = os.path.join(save_dir_batch, "chess_checkpoint_batch_260000.pth")
    print(f"Loading checkpoint from {resume_checkpoint_path}...")
    resume_checkpoint = torch.load(resume_checkpoint_path)
    # Inject the memories!

    model.load_state_dict(resume_checkpoint['model_state_dict'])
    optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])

    start_epoch = resume_checkpoint['epoch']
    start_batch = 260000 # to change later

    print(f"Successfully loaded! Resuming at Epoch {start_epoch}, Batch {start_batch}")
     # --- THE RESUME CODE --- 
    train_model(device, model, dataloader, optimizer, criterion, NUM_EPOCHS, start_batch = start_batch, start_epoch = start_epoch)









    

