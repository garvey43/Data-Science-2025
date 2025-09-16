```markdown
# CodeQuest - Python Text Adventure Game

A text-based RPG adventure game built with Python, now with optional PyGame graphical interface!

## Features

- **Dual Interface**: Choose between classic text-based or new graphical PyGame interface
- Player character with inventory and health system
- Multiple interconnected rooms to explore
- Item collection and usage system (weapons, potions, keys, treasures)
- Puzzle solving challenges with riddles
- Turn-based combat mechanics
- Save/load game functionality
- Graphical user interface with mouse controls (optional)

## How to Play

### Text Version (Default)
```bash
python main.py
```

### Graphical Version (Requires PyGame)
```bash
# First install PyGame:
pip install pygame

# Then run:
python main.py
# Choose option 2 when prompted
```

### Text Commands:
- `go [direction]` - Move in a direction (north, south, east, west)
- `take [item]` - Pick up an item
- `use [item]` - Use an item from inventory
- `inventory` - View your inventory
- `look` - Examine your surroundings
- `attack [target]` - Attack a creature
- `solve [solution]` - Attempt to solve a puzzle
- `save` - Save your game
- `load` - Load a saved game
- `quit` - Exit the game

### Graphical Controls:
- **Mouse Navigation**: Click direction buttons to move
- **Action Buttons**: Click to look, open inventory, use items, or attack
- **Inventory Management**: Click items to use them
- **Keyboard Shortcuts**:
  - `I` - Toggle inventory
  - `ESC` - Close inventory

## Project Structure

```
codequest-adventure-game/
│
├── pygame_ui/           # Graphical interface components
│   ├── __init__.py
│   ├── game_window.py   # Main PyGame window
│   └── ui_elements.py   # UI components (buttons, panels)
│
├── engine/              # Core game systems
│   ├── game_engine.py   # Main game loop and state management
│   ├── command_parser.py # Text command processing
│   └── save_manager.py  # Save/load functionality
│
├── entities/            # Game characters
│   ├── player.py        # Player class with inventory
│   ├── creature.py      # Enemy creatures
│   └── npc.py           # Non-player characters
│
├── world/               # Game world structure
│   ├── room.py          # Room class and management
│   ├── puzzle.py        # Puzzle system
│   └── world_builder.py # World construction
│
├── items/               # Item system
│   ├── consumable.py    # Health potions, etc.
│   ├── weapon.py        # Weapons for combat
│   ├── key.py           # Key items
│   └── treasure.py      # Valuable items
│
├── combat/              # Combat system
│   ├── combat_system.py # Core combat mechanics
│   └── battle.py        # Battle management
│
├── tests/               # Unit tests
│   ├── test_player.py
│   ├── test_combat.py
│   ├── test_items.py
│   └── test_world.py
│
├── utils/               # Utility functions
│   ├── display.py       # Text display utilities
│   ├── helpers.py       # Helper functions
│   └── constants.py     # Game constants
│
├── data/                # Game data
│   └── saves/           # Save game directory
│
├── main.py              # Main entry point (auto-detects interface)
├── requirements.txt     # Dependencies (PyGame)
└── README.md           # This file
```

## Installation

1. **Clone or download the project**
2. **Install dependencies** (optional for graphical version):
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the game**:
   ```bash
   python main.py
   ```

## System Requirements

- **Python 3.6+**
- **PyGame 2.0+** (optional, for graphical interface)
- **Any operating system** (Windows, macOS, Linux)

## Development Features

This project demonstrates:
- **Object-oriented programming** principles
- **SOLID design patterns** and clean architecture
- **Modular code structure** with separation of concerns
- **Dual interface system** (text + graphical)
- **Unit testing** with unittest framework
- **File I/O operations** for save games
- **Event-driven programming** with PyGame
- **UI component design** for game interfaces

## Contributing

Feel free to contribute by:
- Adding new rooms and puzzles
- Creating new item types
- Enhancing the graphical interface
- Improving combat mechanics
- Adding unit tests

## License

This project is open source and available under the MIT License.

## Future Enhancements

Planned features:
- More complex puzzle types
- NPC dialogue system
- Quest system with objectives
- Enhanced graphics with custom sprites
- Sound effects and music
- Multi-language support
- Mobile app version

---

**Enjoy your adventure in CodeQuest!** 🎮
