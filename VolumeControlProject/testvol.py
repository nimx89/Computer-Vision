from __future__ import print_function
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume


def main():
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        if session.Process:
            print(volume.GetMasterVolume(), session.Process.name())
            print("volume.GetMasterVolume(): %s" % volume.GetMasterVolume())
            volume.SetMasterVolume(0.6, None)


if __name__ == "__main__":
    main()