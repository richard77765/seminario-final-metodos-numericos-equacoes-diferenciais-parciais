"""Saída de terminal colorida e consistente (Windows + Linux).

Cada método tem uma cor fixa em todo o projeto:
    MDF -> azul   |   MEF -> verde   |   PINN -> amarelo

As cores são habilitadas automaticamente no console do Windows (VT processing) e
são desligadas sozinhas quando a saída não é um terminal (por exemplo, redirecionada
para arquivo) ou quando a variável de ambiente NO_COLOR está definida. Force com
FORCE_COLOR=1.

Uso típico:
    from src import console as ui
    ui.header("Comparação")
    ui.method_line("MDF", "G_ef = 1.1398")
    ui.ok("tudo certo")
"""

import os
import sys

# --------------------------------------------------------------------------
# Habilitação de cores / UTF-8 (idempotente)
# --------------------------------------------------------------------------

def _enable_windows_vt():
    """Habilita ANSI (Virtual Terminal) no console do Windows 10+. Retorna bool."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        for handle_id in (-11, -12):  # STDOUT, STDERR
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VT
        return True
    except Exception:
        return False


def _enable_utf8():
    """Melhor esforço para saída UTF-8 (para os símbolos renderizarem no Windows)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _supports_color():
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
        return False
    if sys.platform == "win32":
        return _enable_windows_vt()
    return True


_enable_utf8()
COLOR = _supports_color()

# --------------------------------------------------------------------------
# Paleta
# --------------------------------------------------------------------------

_C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "blue": "\033[38;5;39m",     # MDF
    "green": "\033[38;5;42m",    # MEF
    "yellow": "\033[38;5;220m",  # PINN
    "red": "\033[38;5;203m",
    "cyan": "\033[38;5;45m",
    "gray": "\033[38;5;245m",
    "white": "\033[38;5;255m",
}

# Cor fixa por método (o pedido do usuário)
METHOD_COLOR = {"MDF": "blue", "MEF": "green", "PINN": "yellow"}


def paint(text, *styles):
    """Aplica estilos ANSI a ``text`` (no-op se as cores estiverem desligadas)."""
    if not COLOR or not styles:
        return str(text)
    prefix = "".join(_C.get(s, "") for s in styles)
    return f"{prefix}{text}{_C['reset']}"


def method(name):
    """Nome do método na sua cor fixa (MDF azul, MEF verde, PINN amarelo)."""
    return paint(name, METHOD_COLOR.get(name.upper(), "cyan"), "bold")


def chip(name):
    """Rótulo em bloco colorido, ex.:  MDF  (com espaços, em negrito)."""
    color = METHOD_COLOR.get(name.upper(), "cyan")
    return paint(f" {name} ", color, "bold")


# --------------------------------------------------------------------------
# Blocos de saída
# --------------------------------------------------------------------------

_W = 66


def header(title):
    print()
    print(paint("=" * _W, "cyan"))
    print(paint(f"  {title}", "cyan", "bold"))
    print(paint("=" * _W, "cyan"))


def section(title):
    print()
    print(paint(f"-- {title} " + "-" * max(0, _W - len(title) - 4), "gray"))


def rule():
    print(paint("-" * _W, "gray"))


def kv(key, value, key_w=24, key_color="gray"):
    print(f"  {paint(str(key).ljust(key_w), key_color)} {value}")


def method_line(name, value, key_w=6):
    """Linha 'MÉTODO  valor' com o nome na cor do método.

    O preenchimento usa o comprimento do nome SEM cor (os códigos ANSI são
    invisíveis), mantendo o alinhamento correto.
    """
    pad = " " * max(1, key_w - len(name))
    print(f"  {method(name.upper())}{pad}{value}")


def ok(msg):
    print(f"{paint('[OK]', 'green', 'bold')} {msg}")


def warn(msg):
    print(f"{paint('[!]', 'yellow', 'bold')} {msg}")


def err(msg):
    print(f"{paint('[x]', 'red', 'bold')} {msg}")


def info(msg):
    print(f"{paint('*', 'cyan')} {msg}")


def step(msg):
    print(f"{paint('>', 'cyan', 'bold')} {msg}")


def dim(msg):
    print(paint(msg, "dim"))


def value(x, unit=""):
    """Destaca um valor numérico em branco/negrito."""
    return paint(f"{x}{unit}", "white", "bold")


def legend():
    """Imprime a legenda de cores dos métodos."""
    parts = [f"{method(m)}" for m in ("MDF", "MEF", "PINN")]
    print("  " + paint("cores:", "gray") + "  " + "   ".join(parts))
