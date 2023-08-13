from collections import namedtuple
from dataclasses import dataclass, field


class TokenBase:
    def __init_subclass__(cls) -> None:
        def _token_repr(self):
            name = self.__class__.__name__
            return (
                f"Token.{name}({self.value!r})"
                if hasattr(self, "value")
                else f"Token.{name}"
            )

        def _token_eq(self, other):
            return repr(self) == repr(other)

        def _token_ne(self, other):
            return repr(self) != repr(other)

        for name, method in vars(cls).items():
            if name.startswith("__"):
                continue
            method.__repr__ = _token_repr
            method.__eq__ = _token_eq
            method.__ne__ = _token_ne


class Token(TokenBase):
    Ident = namedtuple("Ident", ["value"])
    Int = namedtuple("Int", ["value"])
    Illegal = namedtuple("Illegal", [])
    Eof = namedtuple("Eof", [])
    Assign = namedtuple("Assign", [])
    Bang = namedtuple("Bang", [])
    Dash = namedtuple("Dash", [])
    ForwardSlash = namedtuple("ForwardSlash", [])
    Asterisk = namedtuple("Asterisk", [])
    Equal = namedtuple("Equal", [])
    NotEqual = namedtuple("NotEqual", [])
    LessThan = namedtuple("LessThan", [])
    GreaterThan = namedtuple("GreaterThan", [])
    Plus = namedtuple("Plus", [])
    Comma = namedtuple("Comma", [])
    Semicolon = namedtuple("Semicolon", [])
    Lparen = namedtuple("Lparen", [])
    Rparen = namedtuple("Rparen", [])
    LSquirly = namedtuple("LSquirly", [])
    RSquirly = namedtuple("RSquirly", [])
    Function = namedtuple("Function", [])
    Let = namedtuple("Let", [])
    If = namedtuple("If", [])
    Else = namedtuple("Else", [])
    Return = namedtuple("Return", [])
    True_ = namedtuple("True_", [])
    False_ = namedtuple("False_", [])


@dataclass
class Lexer:
    text: str
    position: int = 0
    read_position: int = 0
    ch: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.read_char()

    def __iter__(self):
        while (token := self.next_token()) != Token.Eof():
            yield token
        yield token

    def next_token(self):
        self.skip_whitespace()

        match self.ch:
            case "{":
                tok = Token.LSquirly()
            case "}":
                tok = Token.RSquirly()
            case "(":
                tok = Token.Lparen()
            case ")":
                tok = Token.Rparen()
            case ",":
                tok = Token.Comma()
            case ";":
                tok = Token.Semicolon()
            case "+":
                tok = Token.Plus()
            case "-":
                tok = Token.Dash()
            case "!":
                if self.peek() == "=":
                    self.read_char()
                    tok = Token.NotEqual()
                else:
                    tok = Token.Bang()
            case ">":
                tok = Token.GreaterThan()
            case "<":
                tok = Token.LessThan()
            case "*":
                tok = Token.Asterisk()
            case "/":
                tok = Token.ForwardSlash()
            case "=":
                if self.peek() == "=":
                    self.read_char()
                    tok = Token.Equal()
                else:
                    tok = Token.Assign()
            case t if t.isalpha() or t == "_":
                match self.read_ident():
                    case "fn":
                        tok = Token.Function()
                    case "let":
                        tok = Token.Let()
                    case "if":
                        tok = Token.If()
                    case "false":
                        tok = Token.False_()
                    case "true":
                        tok = Token.True_()
                    case "return":
                        tok = Token.Return()
                    case "else":
                        tok = Token.Else()
                    case _ as val:
                        tok = Token.Ident(val)
            case t if t.isdigit():
                tok = Token.Int(self.read_int())
            case "":
                tok = Token.Eof()
            case _:
                tok = Token.Illegal()

        self.read_char()
        return tok

    def peek(self) -> str:
        if self.read_position < len(self.text):
            return self.text[self.read_position]
        return ""

    def read_char(self) -> None:
        if self.read_position >= len(self.text):
            self.ch = ""
        else:
            self.ch = self.text[self.read_position]

        self.position = self.read_position
        self.read_position += 1

    def skip_whitespace(self) -> None:
        while self.ch.isspace():
            self.read_char()

    def read_ident(self) -> str:
        pos = self.position
        while self.peek().isalpha() or self.peek() == "_":
            self.read_char()
        return self.text[pos : self.read_position]

    def read_int(self) -> str:
        pos = self.position
        while self.peek().isdigit():
            self.read_char()
        return self.text[pos : self.read_position]
