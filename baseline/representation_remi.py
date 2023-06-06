"""Representation utilities."""
import pathlib
import pprint

import muspy
import numpy as np

import utils

# Configuration
RESOLUTION = 12
# MAX_BEAT = 1024
MAX_DURATION = 384
MAX_BAR = 256

# Duration
KNOWN_DURATIONS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    15,
    16,
    18,
    20,
    21,
    24,
    30,
    36,
    40,
    42,
    48,
    60,
    72,
    84,
    96,
    120,
    144,
    168,
    192,
    384,
]
DURATION_MAP = {
    i: KNOWN_DURATIONS[np.argmin(np.abs(np.array(KNOWN_DURATIONS) - i))]
    for i in range(1, MAX_DURATION + 1)
}

KNOWN_VELOCITIES = {16, 32, 48, 64, 80, 96, 112}

VELOCITY_MAP = {
    i: int(max(1, min(7, round(i / 16))) * 16)
    for i in range(1,128)
}

## Instrument
PROGRAM_INSTRUMENT_MAP = [
    # Pianos
    "grand-piano",
    "bright-piano",
    "electric-grand-piano",
    "honky-tony-piano",
    "electric-piano-1",
    "electric-piano-2",
    "harpsichord",
    "clavinet",
    # Chromatic Percussion
    "celesta",
    "glockenspiel",
    "music-box",
    "vibraphone",
    "marimba",
    "xylophone",
    "tubular-bells",
    "dulcimer",
    # Organs
    "dwawbar-organ",
    "percussive-organ",
    "rock-organ",
    "church-organ",
    "reed-organ",
    "accordion",
    "harmonica",
    "bandoneon",
    # Guitars
    "nylon-string-guitar",
    "steel-string-guitar",
    "jazz-electric-guitar",
    "clean-electric-guitar",
    "muted-electric-guitar",
    "overdriven-electric-guitar",
    "distort-electric-guitar",
    "guitar-harmonic",
    # Basses
    "bass",
    "finger-electric-bass",
    "pick-electric-bass",
    "fretless-electric-bass",
    "slap-bass-1",
    "slap-bass-2",
    "synth-bass-1",
    "synth-bass-2",
    # Strings
    "violin",
    "viola",
    "cello",
    "contrabass",
    "tremelo-strings",
    "pizzicato-strings",
    "harp",
    "timpani",
    # Ensemble
    "strings",
    "strings",
    "synth-strings-1",
    "synth-strings-2",
    "voices-aah",
    "voices-ooh",
    "synth-voice",
    "orchestra-hit",
    # Brass
    "trumpet",
    "trombone",
    "tuba",
    "muted-trumpet",
    "horn",
    "brasses",
    "synth-brasses-1",
    "synth-brasses-2",
    # Reed
    "soprano-saxophone",
    "alto-saxophone",
    "tenor-saxophone",
    "baritone-saxophone",
    "oboe",
    "english-horn",
    "bassoon",
    "clarinet",
    # Pipe
    "piccolo",
    "flute",
    "recorder",
    "pan-flute",
    "blown-bottle",
    "Shakuhachi",
    "Whistle",
    "ocarina",
    # Synth Lead
    "lead-square",
    "lead-sawtooth",
    "lead-calliope",
    "lead-chiff",
    "lead-charang",
    "lead-voice",
    "lead-fifths",
    "lead-bass+lead",
    # Synth Pad
    "pad-new-age",
    "pad-warm",
    "pad-polysynth",
    "pad-choir",
    "pad-bowed",
    "pad-metallic",
    "pad-halo",
    "pad-sweep",
    # Synth Effects
    "fx-rain",
    "fx-soundtrack",
    "fx-crystal",
    "fx-atmosphere",
    "fx-brightness",
    "fx-goblins",
    "fx-echoes",
    "fx-scifi",
    # Ethnic
    "sitar",
    "banjo",
    "shamisen",
    "koto",
    "kalimba",
    "bag-pipe",
    "violin",
    "shehnai",
    # Percussive
    "tinkle-bell",
    "agogo",
    "steel-drum",
    "woodblock",
    "taiko",
    "melodic-tom",
    "synth-drums",
    "reverse-cymbal",
    "guitar-fret-noise",
    # Sound effects
    "breath-noise",
    "seashore",
    "bird-tweet",
    "telephone-rang",
    "helicopter",
    "applause",
    "gunshot",
    "drumset",
]
INSTRUMENT_PROGRAM_MAP = {
    instrument: program
    for program, instrument in enumerate(PROGRAM_INSTRUMENT_MAP)
}
KNOWN_PROGRAMS = list(
    k for k, v in INSTRUMENT_PROGRAM_MAP.items() if v is not None
)
KNOWN_INSTRUMENTS = list(dict.fromkeys(INSTRUMENT_PROGRAM_MAP.keys()))

KNOWN_EVENTS = [
    "start-of-song",
    "end-of-song",
]
KNOWN_EVENTS.extend(f"bar_{i}" for i in range(MAX_BAR))
KNOWN_EVENTS.extend(f"time-signature_{i}" for i in (3, 4))
KNOWN_EVENTS.extend(f"position_{i}" for i in range(4*RESOLUTION))
KNOWN_EVENTS.extend(
    f"instrument_{instrument}" for instrument in KNOWN_INSTRUMENTS
)
KNOWN_EVENTS.extend(f"pitch_{i}" for i in range(128))
KNOWN_EVENTS.extend(f"duration_{i}" for i in KNOWN_DURATIONS)
KNOWN_EVENTS.extend(f"velocity_{i}" for i in KNOWN_VELOCITIES)
EVENT_CODE_MAPS = {event: i for i, event in enumerate(KNOWN_EVENTS)}
CODE_EVENT_MAPS = utils.inverse_dict(EVENT_CODE_MAPS)


class Indexer:
    def __init__(self, data=None, is_training=False):
        self._dict = dict() if data is None else data
        self._is_training = is_training

    def __getitem__(self, key):
        if self._is_training and key not in self._dict:
            self._dict[key] = len(self._dict)
            return len(self._dict) - 1
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __contain__(self, item):
        return item in self._dict

    def get_dict(self):
        """Return the internal dictionary."""
        return self._dict

    def train(self):
        """Set training mode."""
        self._is_training = True

    def eval(self):
        """Set evaluation mode."""
        self._is_learning = False


def get_encoding():
    """Return the encoding configurations."""
    return {
        "resolution": RESOLUTION,
        "max_beat": MAX_BEAT,
        "max_duration": MAX_DURATION,
        "program_instrument_map": PROGRAM_INSTRUMENT_MAP,
        "instrument_program_map": INSTRUMENT_PROGRAM_MAP,
        "duration_map": DURATION_MAP,
        "velocity_map": VELOCITY_MAP,
        "event_code_map": EVENT_CODE_MAPS,
        "code_event_map": CODE_EVENT_MAPS,
    }


def load_encoding(filename):
    """Load encoding configurations from a JSON file."""
    encoding = utils.load_json(filename)
    for key in ("program_instrument_map", "code_event_map", "duration_map", "velocity_map"):
        encoding[key] = {
            int(k) if k != "null" else None: v
            for k, v in encoding[key].items()
        }
    return encoding


def extract_notes(music, resolution):
    """Return a MusPy music object as a note sequence.

    Each row of the output is a note specified as follows.

        (onset, pitch, duration, program, velocity)

    """
    # Check resolution
    assert music.resolution == resolution
    assert all([(ts.numerator == 3 or ts.numerator == 4) and ts.denominator == 4 for ts in music.time_signatures])

    # Extract notes
    notes = []
    for track in music:
        for note in track:
            beat, position = divmod(note.time, resolution)
            notes.append(
                (note.time, note.pitch, note.duration, track.program, note.velocity)
            )

    # Deduplicate and sort the notes
    notes = sorted(set(notes))

    return np.array(notes)


def encode_notes(notes, encoding, indexer):
    """Encode the notes into a sequence of code tuples.

    Each row of the output is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    # Get variables
    max_beat = encoding["max_beat"]
    max_duration = encoding["max_duration"]

    # Get maps
    duration_map = encoding["duration_map"]
    program_instrument_map = encoding["program_instrument_map"]
    velocity_map = encoding["velocity_map"]

    # Start the codes with an SOS event
    codes = [indexer["start-of-song"]]

    # Encode the notes
    last_beat = 0
    for beat, position, pitch, duration, program, velocity in notes:
        # Skip if max_beat has reached
        if beat > max_beat:
            continue
        # Skip unknown instruments
        instrument = program_instrument_map[program]
        if instrument is None:
            continue
        if beat > last_beat:
            codes.append(indexer[f"beat_{beat}"])
            last_beat = beat
        codes.append(indexer[f"position_{position}"])
        codes.append(indexer[f"instrument_{instrument}"])
        codes.append(indexer[f"pitch_{pitch}"])
        codes.append(
            indexer[f"duration_{duration_map[min(duration, max_duration)]}"]
        )
        codes.append(indexer[f"velocity_{velocity_map[velocity]}"])

    # End the codes with an EOS event
    codes.append(indexer["end-of-song"])

    return np.array(codes)


def encode(music, encoding, indexer):
    """Encode a MusPy music object into a sequence of codes.

    Each row of the input is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    # Extract notes
    notes = extract_notes(music, encoding["resolution"])

    # Encode the notes
    codes = encode_notes(notes, encoding, indexer)

    return codes


def decode_notes(data, encoding, vocabulary):
    """Decode codes into a note sequence."""
    # Get variables and maps
    instrument_program_map = encoding["instrument_program_map"]

    # Initialize variables
    beat = 0
    position = None
    program = None
    pitch = None
    duration = None
    velocity = None

    # Decode the codes into a sequence of notes
    notes = []
    for code in data:
        event = vocabulary[code]
        if event == "start-of-song":
            continue
        elif event == "end-of-song":
            break
        elif event.startswith("beat"):
            beat = int(event.split("_")[1])
            # Reset variables
            position = None
            program = None
            pitch = None
            duration = None
        elif event.startswith("position"):
            position = int(event.split("_")[1])
            # Reset variables
            program = None
            pitch = None
            duration = None
        elif event.startswith("instrument"):
            instrument = event.split("_")[1]
            program = instrument_program_map[instrument]
        elif event.startswith("pitch"):
            pitch = int(event.split("_")[1])
        elif event.startswith("duration"):
            duration = int(event.split("_")[1])
        elif event.startswith("velocity"):
            velocity = int(event.split("_")[1])
            if (
                position is None
                or program is None
                or pitch is None
                or duration is None
            ):
                continue
            notes.append((beat, position, pitch, duration, program, velocity))
        else:
            raise ValueError(f"Unknown event type for: {event}")

    return notes


def reconstruct(notes, resolution):
    """Reconstruct a note sequence to a MusPy Music object."""
    # Construct the MusPy Music object
    music = muspy.Music(resolution=resolution, tempos=[muspy.Tempo(0, 100)])

    # Append the tracks
    programs = sorted(set(note[-2] for note in notes))
    for program in programs:
        music.tracks.append(muspy.Track(program))

    # Append the notes
    for beat, position, pitch, duration, program, velocity in notes:
        time = beat * resolution + position
        track_idx = programs.index(program)
        music[track_idx].notes.append(muspy.Note(time, pitch, duration, velocity))

    return music


def decode(codes, encoding, vocabulary):
    """Decode codes into a MusPy Music object.

    Each row of the input is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    # Get resolution
    resolution = encoding["resolution"]

    # Decode codes into a note sequence
    notes = decode_notes(codes, encoding, vocabulary)

    # Reconstruct the music object
    music = reconstruct(notes, resolution)

    return music


def dump(data, vocabulary):
    """Decode the codes and dump as a string."""
    # Iterate over the rows
    lines = []
    for code in data:
        event = vocabulary[code]
        if (
            event == "start-of-song"
            or event.startswith("beat")
            or event.startswith("position")
        ):
            lines.append(event)
        elif event == "end-of-song":
            lines.append(event)
            break
        elif (
            event.startswith("instrument")
            or event.startswith("pitch")
            or event.startswith("duration")
            or event.startswith("velocity")
        ):
            lines[-1] = f"{lines[-1]} {event}"
        else:
            raise ValueError(f"Unknown event type for: {event}")

    return "\n".join(lines)


def save_txt(filename, data, vocabulary):
    """Dump the codes into a TXT file."""
    with open(filename, "w") as f:
        f.write(dump(data, vocabulary))


def save_csv_notes(filename, data):
    """Save the representation as a CSV file."""
    assert data.shape[1] == 5
    np.savetxt(
        filename,
        data,
        fmt="%d",
        delimiter=",",
        header="beat,position,pitch,duration,program,velocity",
        comments="",
    )


def save_csv_codes(filename, data):
    """Save the representation as a CSV file."""
    assert data.ndim == 1
    np.savetxt(
        filename,
        data,
        fmt="%d",
        delimiter=",",
        header="code",
        comments="",
    )


def main():
    """Main function."""
    # Get the encoding
    encoding = get_encoding()

    # Save the encoding
    filename = pathlib.Path(__file__).parent / "encoding_remi.json"
    utils.save_json(filename, encoding)

    # Load encoding
    encoding = load_encoding(filename)

    # Print the maps
    print(f"{' Maps ':=^40}")
    for key, value in encoding.items():
        if key in ("program_instrument_map", "instrument_program_map"):
            print("-" * 40)
            print(f"{key}:")
            pprint.pprint(value, indent=2)

    # Print the variables
    print(f"{' Variables ':=^40}")
    print(f"resolution: {encoding['resolution']}")
    print(f"max_beat: {encoding['max_beat']}")
    print(f"max_duration: {encoding['max_duration']}")

    # Load the example
    music = muspy.load(pathlib.Path(__file__).parent / "example.json")

    # Get the indexer
    indexer = Indexer(is_training=True)

    # Encode the music
    encoded = encode(music, encoding, indexer)
    print(f"Codes:\n{encoded}")

    # Get the learned vocabulary
    vocabulary = utils.inverse_dict(indexer.get_dict())

    print("-" * 40)
    print(f"Decoded:\n{dump(encoded, vocabulary)}")

    music = decode(encoded, encoding, vocabulary)
    print(f"Decoded musics:\n{music}")


if __name__ == "__main__":
    main()
