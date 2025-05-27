"""Terminal color utilities for better visualization."""


class Colors:
    """ANSI color codes for terminal output formatting."""
    
    # Text colors
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    @classmethod
    def colored_text(cls, text: str, color: str) -> str:
        """Return colored text."""
        return f"{color}{text}{cls.ENDC}"
    
    @classmethod
    def success(cls, text: str) -> str:
        """Return green colored text for success messages."""
        return cls.colored_text(text, cls.GREEN)
    
    @classmethod
    def warning(cls, text: str) -> str:
        """Return yellow colored text for warning messages."""
        return cls.colored_text(text, cls.WARNING)
    
    @classmethod
    def error(cls, text: str) -> str:
        """Return red colored text for error messages."""
        return cls.colored_text(text, cls.FAIL)
    
    @classmethod
    def info(cls, text: str) -> str:
        """Return cyan colored text for info messages."""
        return cls.colored_text(text, cls.CYAN)
    
    @classmethod
    def header(cls, text: str) -> str:
        """Return header formatted text."""
        return cls.colored_text(text, cls.HEADER + cls.BOLD)


def print_header(title: str, width: int = 80) -> None:
    """Print a formatted header for better visual separation."""
    print("\n" + "=" * width)
    print(f"{Colors.BOLD}{Colors.HEADER}{title.center(width)}{Colors.ENDC}")
    print("=" * width + "\n")


def print_section(title: str, content: str = "") -> None:
    """Print a formatted section."""
    print(f"\n{Colors.BOLD}{title}:{Colors.ENDC}")
    if content:
        print(content)
