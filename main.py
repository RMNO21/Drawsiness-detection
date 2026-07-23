"""Drowsiness Detection System - Main Entry Point."""
import sys, signal, logging
from PySide6.QtWidgets import QApplication
from config import AppConfig
from logger import setup_logging
from gui import DrowsinessApp

def _sigint(signum, frame):
    logging.info("Signal %d received", signum); sys.exit(0)

def main():
    signal.signal(signal.SIGINT, _sigint)
    config = AppConfig.load()
    setup_logging(config.log_level, config.log_dir)
    logging.info("="*60)
    logging.info("Drowsiness Detection System v1.0.0")
    logging.info("="*60)
    qapp = QApplication(sys.argv)
    app = DrowsinessApp(config)
    app.show()
    return qapp.exec()

if __name__ == "__main__":
    sys.exit(main())
