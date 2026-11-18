"""Sound clips Loopy plays (not speech), e.g. when someone hits the laptop.

    clip = Clip(HO_YEAH)
    clip.finished.connect(...)
    clip.play()          # False if it's already playing
"""

from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

DATA = Path(__file__).parent / "data"
HO_YEAH = DATA / "can-i-get-a-ho-yeah.mp3"


class Clip(QObject):
    finished = Signal()

    def __init__(self, path: Path):
        super().__init__()
        self._player = QMediaPlayer(self)
        self._output = QAudioOutput(self)
        self._player.setAudioOutput(self._output)
        self._player.setSource(QUrl.fromLocalFile(str(path)))
        self._player.playbackStateChanged.connect(self._state_changed)

    @property
    def playing(self) -> bool:
        return self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def play(self) -> bool:
        if self.playing:
            return False
        self._player.setPosition(0)
        self._player.play()
        return True

    def stop(self) -> None:
        self._player.stop()

    def _state_changed(self, state) -> None:
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self.finished.emit()
