"""Representation utilities."""
import pathlib
import pprint
from collections import defaultdict

import muspy
import numpy as np

import utils

# Configuration
RESOLUTION = 12
# MAX_BEAT = 1024
MAX_TIME_SHIFT = RESOLUTION * 4
MAX_BAR = 8
MAX_TRACK_NUM = 12

# Instrument
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
PROGRAM_INSTRUMENT_MAP = {
    program: instrument
    for program, instrument in enumerate(PROGRAM_INSTRUMENT_MAP)
}
KNOWN_PROGRAMS = list(
    k for k, v in INSTRUMENT_PROGRAM_MAP.items() if v is not None
)
KNOWN_INSTRUMENTS = list(dict.fromkeys(INSTRUMENT_PROGRAM_MAP.keys()))

KNOWN_EVENTS = [
    "start-of-song",
    "end-of-song",
    "start-of-bar",
    "end-of-bar",
    "start-of-track",
    "end-of-track",
]
KNOWN_EVENTS.extend(
    f"instrument_{instrument}" for instrument in KNOWN_INSTRUMENTS
)
KNOWN_EVENTS.extend(f"note-on_{i}" for i in range(128))
KNOWN_EVENTS.extend(f"note-off_{i}" for i in range(128))
KNOWN_EVENTS.extend(f"time-shift_{i}" for i in range(1, MAX_TIME_SHIFT + 1))
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
        "max_bar": MAX_BAR,
        'max_track_num': MAX_TRACK_NUM,
        "max_time_shift": MAX_TIME_SHIFT,
        "program_instrument_map": PROGRAM_INSTRUMENT_MAP,
        "instrument_program_map": INSTRUMENT_PROGRAM_MAP,
        "event_code_map": EVENT_CODE_MAPS,
        "code_event_map": CODE_EVENT_MAPS,
    }


def load_encoding(filename):
    """Load encoding configurations from a JSON file."""
    encoding = utils.load_json(filename)
    for key in ("program_instrument_map", "code_event_map"):
        encoding[key] = {
            int(k) if k != "null" else None: v
            for k, v in encoding[key].items()
        }
    return encoding


def extract_notes(music, resolution):
    """Return a MusPy music object as a note sequence.

    Each row of the output is a note specified as follows.

        (beat, position, pitch, duration, program)

    """
    # Check resolution
    assert music.resolution == resolution

    # Extract notes
    notes = []
    for track in music:
        if track.program not in KNOWN_PROGRAMS:
            continue
        for note in track:
            beat, position = divmod(note.time, resolution)
            notes.append(
                (beat, position, note.pitch, note.duration, track.program)
            )

    # Deduplicate and sort the notes
    notes = sorted(set(notes))

    return np.array(notes)


# def encode_notes(notes, encoding, indexer):
#     """Encode the notes into a sequence of code tuples.

#     Each row of the output is encoded as follows.

#         (event_type, beat, position, pitch, duration, instrument)

#     """
#     # Get variables
#     resolution = encoding["resolution"]
#     max_beat = encoding["max_beat"]
#     max_time_shift = encoding["max_time_shift"]

#     # Get maps
#     program_instrument_map = encoding["program_instrument_map"]
#     instrument_program_map = encoding["instrument_program_map"]

#     # Extract notes
#     instruments = defaultdict(list)
#     for note in notes:
#         instrument = program_instrument_map[note[-1]]
#         # Skip unknown instruments
#         if instrument is None:
#             continue
#         instruments[instrument].append(note)

#     # Sort the instruments
#     instruments = dict(
#         sorted(
#             instruments.items(),
#             key=lambda x: instrument_program_map[x[0]],
#         )
#     )

#     # Collect events
#     events = defaultdict(list)
#     for instrument, instrument_notes in instruments.items():
#         for beat, position, pitch, duration, _ in instrument_notes:
#             if beat > max_beat:
#                 continue
#             time = beat * resolution + position
#             events[instrument].append((time, f"note-on_{pitch}"))
#             events[instrument].append((time + duration, f"note-off_{pitch}"))

#     # Deduplicate and sort the events
#     for instrument in events:
#         events[instrument] = sorted(set(events[instrument]))

#     # Start the codes with an SOS event
#     codes = [indexer["start-of-song"]]

#     # Encode the instruments
#     for instrument in events:
#         codes.append(indexer["start-of-track"])
#         codes.append(indexer[f"instrument_{instrument}"])
#         time = 0
#         for event_time, event in events[instrument]:
#             while time < event_time:
#                 time_shift = min(event_time - time, max_time_shift)
#                 codes.append(indexer[f"time-shift_{time_shift}"])
#                 time += time_shift
#             codes.append(indexer[event])
#         codes.append(indexer["end-of-track"])

#     # End the codes with an EOS event
#     codes.append(indexer["end-of-song"])

#     return np.array(codes)


def track_list_to_code(track_list, indexer):
    np.random.shuffle(track_list)
    codes = [indexer['start-of-song']]
    for track in track_list:
        codes.extend(track)
    codes.append(indexer['end-of-song'])
    return np.array(codes, dtype=np.int32)


def encode(music, encoding, indexer):
    """Encode a MusPy music object into a sequence of codes.

    Each row of the input is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    # Extract notes
    # notes = extract_notes(music, encoding["resolution"])

    # # Encode the notes
    # codes = encode_notes(notes, encoding, indexer)

    assert music.resolution == encoding['resolution']

    assert len(music.tracks) <= encoding['max_track_num']

    bar_length = encoding['resolution'] * 4
    max_onset = encoding['max_bar'] * bar_length
    assert all([
        ts.numerator == 4 and ts.denominator == 4
        for ts in music.time_signatures
        if ts.time < max_onset
    ])

    if len(music.time_signatures) != 0:
        assert music.time_signatures[0].time == 0

    sot_code = indexer['start-of-track']
    eot_code = indexer['end-of-track']
    sob_code = indexer['start-of-bar']
    eob_code = indexer['end-of-bar']

    track_list = []
    for track in music:
        program = 128 if track.is_drum else track.program
        instrument = encoding["program_instrument_map"][program]
        cur_track = [sot_code, indexer[f'instrument_{instrument}'], sob_code]

        note_event_list = []
        for note in track:
            if note.time < max_onset:
                note_event_list.append((note.time, f'note-on_{note.pitch}'))
                note_event_list.append((note.time+note.duration, f'note-off_{note.pitch}'))
        # note_event_list = sorted(set(note_event_list))
        note_event_list.sort()

        next_bar_start_time = bar_length
        note_cursor = 0
        prev_time = 0
        while note_cursor < len(note_event_list):
            note_event = note_event_list[note_cursor]
            if note_event[0] >= next_bar_start_time:
                cur_track.append(eob_code)
                cur_track.append(sob_code)
                next_bar_start_time += bar_length
                prev_time += bar_length
            else:
                if note_event[0] > prev_time:
                    # if note_event[0] - prev_time > bar_length:
                    #     print(note_event[0], prev_time, next_bar_start_time)
                    #     raise ValueError
                    cur_track.append(indexer[f'time-shift_{note_event[0] - prev_time}'])
                    prev_time = note_event[0]
                cur_track.append(indexer[note_event[1]])
                note_cursor += 1

        cur_track.extend([eob_code, eot_code])
        track_list.append(cur_track)

    return track_list


def decode_notes(data, encoding, vocabulary):
    """Decode codes into a note sequence."""
    # Get variables and maps
    # resolution = encoding["resolution"]
    instrument_program_map = encoding["instrument_program_map"]

    # Initialize variables
    program = 0
    bar_start_time = -encoding['resolution'] * 4
    time = 0
    note_ons = {}

    # Decode the codes into a sequence of notes
    notes = []
    for code in data:
        event = vocabulary[code]
        if event == "start-of-song":
            continue
        elif event == "end-of-song":
            break
        elif event in ("start-of-track", "end-of-track"):
            # Reset variables
            program = 0
            time = 0
            note_ons = {}
        elif event == "start-of-bar":
            bar_start_time = bar_start_time + encoding['resolution'] * 4
            time = bar_start_time
        elif event == "end-of-bar":
            continue
        elif event.startswith("instrument"):
            instrument = event.split("_")[1]
            program = instrument_program_map[instrument]
        elif event.startswith("time-shift"):
            time += int(event.split("_")[1])
        elif event.startswith("note-on"):
            pitch = int(event.split("_")[1])
            note_ons[pitch] = time
        elif event.startswith("note-off"):
            pitch = int(event.split("_")[1])
            # Skip a note-off event without a corresponding note-on event
            if pitch not in note_ons:
                continue
            onset = note_ons[pitch]
            notes.append(
                (onset, pitch, time - note_ons[pitch], program)
            )
        else:
            raise ValueError(f"Unknown event type for: {event}")

    return notes


def reconstruct(notes, resolution):
    """Reconstruct a note sequence to a MusPy Music object."""
    # Construct the MusPy Music object
    music = muspy.Music(resolution=resolution, tempos=[muspy.Tempo(0, 100)])

    # Append the tracks
    programs = sorted(set(note[-1] for note in notes))
    for program in programs:
        if program == 128:
            music.tracks.append(muspy.Track(is_drum=True))
        else:
            music.tracks.append(muspy.Track(program))

    # Append the notes
    for onset, pitch, duration, program in notes:
        track_idx = programs.index(program)
        music[track_idx].notes.append(muspy.Note(onset, pitch, duration))

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
            event in ("start-of-song", "start-of-track", "end-of-track", "start-of-bar", "end-of-bar")
            or event.startswith("instrument")
            or event.startswith("time-shift")
            or event.startswith("note-on")
            or event.startswith("note-off")
        ):
            lines.append(event)
        elif event == "end-of-song":
            lines.append(event)
            break
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
        header="beat,position,pitch,duration,program",
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
    filename = pathlib.Path(__file__).parent / "encoding_mmm.json"
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
    print(f"max_bar: {encoding['max_bar']}")
    print(f"max_time_shift: {encoding['max_time_shift']}")

    # Load the example
    music = muspy.load(pathlib.Path(__file__).parent / "example_mmm.json")

    # Get the indexer
    indexer = Indexer(is_training=True)

    # Encode the music
    track_list = encode(music, encoding, indexer)
    encoded = track_list_to_code(track_list, indexer)
    print(f"Codes:\n{encoded}")

    # Get the learned vocabulary
    vocabulary = utils.inverse_dict(indexer.get_dict())

    print("-" * 40)
    print(f"Decoded:\n{dump(encoded, vocabulary)}")

    music = decode(encoded, encoding, vocabulary)
    print(f"Decoded musics:\n{music}")


if __name__ == "__main__":
    main()
