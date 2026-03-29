#!/usr/bin/env python3
"""
Build scripture cross-reference data for the arc diagram visualization.
Processes OpenBible.info cross-references and adds Bible-Quran connections.
"""

import json
import re
from collections import defaultdict

# ── Bible book metadata ──────────────────────────────────────────────
# Abbreviation → (full name, testament, chapter_count, total_verses)
BIBLE_BOOKS = [
    ("Gen", "Genesis", "OT", 50, 1533),
    ("Exod", "Exodus", "OT", 40, 1213),
    ("Lev", "Leviticus", "OT", 27, 859),
    ("Num", "Numbers", "OT", 36, 1288),
    ("Deut", "Deuteronomy", "OT", 34, 959),
    ("Josh", "Joshua", "OT", 24, 658),
    ("Judg", "Judges", "OT", 21, 618),
    ("Ruth", "Ruth", "OT", 4, 85),
    ("1Sam", "1 Samuel", "OT", 31, 810),
    ("2Sam", "2 Samuel", "OT", 24, 695),
    ("1Kgs", "1 Kings", "OT", 22, 816),
    ("2Kgs", "2 Kings", "OT", 25, 719),
    ("1Chr", "1 Chronicles", "OT", 29, 942),
    ("2Chr", "2 Chronicles", "OT", 36, 822),
    ("Ezra", "Ezra", "OT", 10, 280),
    ("Neh", "Nehemiah", "OT", 13, 406),
    ("Esth", "Esther", "OT", 10, 167),
    ("Job", "Job", "OT", 42, 1070),
    ("Ps", "Psalms", "OT", 150, 2461),
    ("Prov", "Proverbs", "OT", 31, 915),
    ("Eccl", "Ecclesiastes", "OT", 12, 222),
    ("Song", "Song of Solomon", "OT", 8, 117),
    ("Isa", "Isaiah", "OT", 66, 1292),
    ("Jer", "Jeremiah", "OT", 52, 1364),
    ("Lam", "Lamentations", "OT", 5, 154),
    ("Ezek", "Ezekiel", "OT", 48, 1273),
    ("Dan", "Daniel", "OT", 12, 357),
    ("Hos", "Hosea", "OT", 14, 197),
    ("Joel", "Joel", "OT", 3, 73),
    ("Amos", "Amos", "OT", 9, 146),
    ("Obad", "Obadiah", "OT", 1, 21),
    ("Jonah", "Jonah", "OT", 4, 48),
    ("Mic", "Micah", "OT", 7, 105),
    ("Nah", "Nahum", "OT", 3, 47),
    ("Hab", "Habakkuk", "OT", 3, 56),
    ("Zeph", "Zephaniah", "OT", 3, 53),
    ("Hag", "Haggai", "OT", 2, 38),
    ("Zech", "Zechariah", "OT", 14, 211),
    ("Mal", "Malachi", "OT", 4, 55),
    ("Matt", "Matthew", "NT", 28, 1071),
    ("Mark", "Mark", "NT", 16, 678),
    ("Luke", "Luke", "NT", 24, 1151),
    ("John", "John", "NT", 21, 879),
    ("Acts", "Acts", "NT", 28, 1007),
    ("Rom", "Romans", "NT", 16, 433),
    ("1Cor", "1 Corinthians", "NT", 16, 437),
    ("2Cor", "2 Corinthians", "NT", 13, 257),
    ("Gal", "Galatians", "NT", 6, 149),
    ("Eph", "Ephesians", "NT", 6, 155),
    ("Phil", "Philippians", "NT", 4, 104),
    ("Col", "Colossians", "NT", 4, 95),
    ("1Thess", "1 Thessalonians", "NT", 5, 89),
    ("2Thess", "2 Thessalonians", "NT", 3, 47),
    ("1Tim", "1 Timothy", "NT", 6, 113),
    ("2Tim", "2 Timothy", "NT", 4, 83),
    ("Titus", "Titus", "NT", 3, 46),
    ("Phlm", "Philemon", "NT", 1, 25),
    ("Heb", "Hebrews", "NT", 13, 303),
    ("Jas", "James", "NT", 5, 108),
    ("1Pet", "1 Peter", "NT", 5, 105),
    ("2Pet", "2 Peter", "NT", 3, 61),
    ("1John", "1 John", "NT", 5, 105),
    ("2John", "2 John", "NT", 1, 13),
    ("3John", "3 John", "NT", 1, 15),
    ("Jude", "Jude", "NT", 1, 25),
    ("Rev", "Revelation", "NT", 22, 404),
]

TOTAL_BIBLE_VERSES = sum(b[4] for b in BIBLE_BOOKS)

# Build lookup: abbreviation → index
ABBR_TO_IDX = {}
for i, (abbr, name, testament, chapters, verses) in enumerate(BIBLE_BOOKS):
    ABBR_TO_IDX[abbr] = i

# ── Quran surah metadata ─────────────────────────────────────────────
QURAN_SURAHS = [
    (1, "Al-Fatihah", 7), (2, "Al-Baqarah", 286), (3, "Ali 'Imran", 200),
    (4, "An-Nisa", 176), (5, "Al-Ma'idah", 120), (6, "Al-An'am", 165),
    (7, "Al-A'raf", 206), (8, "Al-Anfal", 75), (9, "At-Tawbah", 129),
    (10, "Yunus", 109), (11, "Hud", 123), (12, "Yusuf", 111),
    (13, "Ar-Ra'd", 43), (14, "Ibrahim", 52), (15, "Al-Hijr", 99),
    (16, "An-Nahl", 128), (17, "Al-Isra", 111), (18, "Al-Kahf", 110),
    (19, "Maryam", 98), (20, "Ta-Ha", 135), (21, "Al-Anbiya", 112),
    (22, "Al-Hajj", 78), (23, "Al-Mu'minun", 118), (24, "An-Nur", 64),
    (25, "Al-Furqan", 77), (26, "Ash-Shu'ara", 227), (27, "An-Naml", 93),
    (28, "Al-Qasas", 88), (29, "Al-Ankabut", 69), (30, "Ar-Rum", 60),
    (31, "Luqman", 34), (32, "As-Sajdah", 30), (33, "Al-Ahzab", 73),
    (34, "Saba", 54), (35, "Fatir", 45), (36, "Ya-Sin", 83),
    (37, "As-Saffat", 182), (38, "Sad", 88), (39, "Az-Zumar", 75),
    (40, "Ghafir", 85), (41, "Fussilat", 54), (42, "Ash-Shura", 53),
    (43, "Az-Zukhruf", 89), (44, "Ad-Dukhan", 59), (45, "Al-Jathiyah", 37),
    (46, "Al-Ahqaf", 35), (47, "Muhammad", 38), (48, "Al-Fath", 29),
    (49, "Al-Hujurat", 18), (50, "Qaf", 45), (51, "Adh-Dhariyat", 60),
    (52, "At-Tur", 49), (53, "An-Najm", 62), (54, "Al-Qamar", 55),
    (55, "Ar-Rahman", 78), (56, "Al-Waqi'ah", 96), (57, "Al-Hadid", 29),
    (58, "Al-Mujadila", 22), (59, "Al-Hashr", 24), (60, "Al-Mumtahanah", 13),
    (61, "As-Saff", 14), (62, "Al-Jumu'ah", 11), (63, "Al-Munafiqun", 11),
    (64, "At-Taghabun", 18), (65, "At-Talaq", 12), (66, "At-Tahrim", 12),
    (67, "Al-Mulk", 30), (68, "Al-Qalam", 52), (69, "Al-Haqqah", 52),
    (70, "Al-Ma'arij", 44), (71, "Nuh", 28), (72, "Al-Jinn", 28),
    (73, "Al-Muzzammil", 20), (74, "Al-Muddaththir", 56), (75, "Al-Qiyamah", 40),
    (76, "Al-Insan", 31), (77, "Al-Mursalat", 50), (78, "An-Naba", 40),
    (79, "An-Nazi'at", 46), (80, "Abasa", 42), (81, "At-Takwir", 29),
    (82, "Al-Infitar", 19), (83, "Al-Mutaffifin", 36), (84, "Al-Inshiqaq", 25),
    (85, "Al-Buruj", 22), (86, "At-Tariq", 17), (87, "Al-A'la", 19),
    (88, "Al-Ghashiyah", 26), (89, "Al-Fajr", 30), (90, "Al-Balad", 20),
    (91, "Ash-Shams", 15), (92, "Al-Layl", 21), (93, "Ad-Duhaa", 11),
    (94, "Ash-Sharh", 8), (95, "At-Tin", 8), (96, "Al-Alaq", 19),
    (97, "Al-Qadr", 5), (98, "Al-Bayyinah", 8), (99, "Az-Zalzalah", 8),
    (100, "Al-Adiyat", 11), (101, "Al-Qari'ah", 11), (102, "At-Takathur", 8),
    (103, "Al-Asr", 3), (104, "Al-Humazah", 9), (105, "Al-Fil", 5),
    (106, "Quraysh", 4), (107, "Al-Ma'un", 7), (108, "Al-Kawthar", 3),
    (109, "Al-Kafirun", 6), (110, "An-Nasr", 3), (111, "Al-Masad", 5),
    (112, "Al-Ikhlas", 4), (113, "Al-Falaq", 5), (114, "An-Nas", 6),
]

TOTAL_QURAN_VERSES = sum(s[2] for s in QURAN_SURAHS)

# ── Bible-Quran theological connections ───────────────────────────────
# These represent well-documented parallels between Bible and Quran passages.
# Sources: Gabriel Said Reynolds, Sidney Griffith, Reza Shah-Kazemi,
# Michel Cuypers, and scholarly cross-referencing.
#
# Format: (bible_ref, quran_surah, quran_ayah, theme, strength)
# strength: 1-3 (1=thematic parallel, 2=narrative parallel, 3=direct reference)

BIBLE_QURAN_CONNECTIONS = [
    # ── CREATION NARRATIVES ──
    ("Gen.1.1", 2, 29, "creation", 3),       # "Created the heavens and the earth"
    ("Gen.1.1", 6, 1, "creation", 3),
    ("Gen.1.1", 21, 30, "creation", 3),       # heavens and earth joined then split
    ("Gen.1.26", 2, 30, "creation_of_adam", 3),  # creation of Adam / khalifah
    ("Gen.1.26", 15, 28, "creation_of_adam", 3),
    ("Gen.2.7", 15, 29, "breath_of_life", 3),    # God breathes into Adam
    ("Gen.2.7", 38, 72, "breath_of_life", 3),
    ("Gen.2.19", 2, 31, "naming_animals", 3),     # Adam names things
    ("Gen.2.35", 2, 35, "garden_eden", 3),
    ("Gen.3.1", 7, 20, "temptation", 3),          # Serpent/Iblis temptation
    ("Gen.3.1", 20, 120, "temptation", 3),
    ("Gen.3.7", 7, 22, "fall_shame", 3),           # shame/covering
    ("Gen.3.23", 2, 36, "expulsion", 3),           # expelled from garden
    ("Gen.3.23", 7, 24, "expulsion", 3),

    # ── CAIN AND ABEL ──
    ("Gen.4.1", 5, 27, "cain_abel", 3),            # story of two sons
    ("Gen.4.8", 5, 30, "murder_brother", 3),
    ("Gen.4.10", 5, 32, "killing_one_soul", 3),    # killing one = killing all

    # ── NOAH AND THE FLOOD ──
    ("Gen.6.5", 11, 25, "noah_mission", 3),
    ("Gen.6.14", 11, 37, "build_ark", 3),
    ("Gen.6.14", 23, 27, "build_ark", 3),
    ("Gen.7.11", 11, 40, "flood_begins", 3),
    ("Gen.7.11", 54, 11, "flood_begins", 3),
    ("Gen.8.4", 11, 44, "ark_rests", 3),           # Mt Ararat / Al-Judi
    ("Gen.9.12", 11, 48, "covenant_after_flood", 2),
    ("Gen.11.1", 71, 1, "noah_surah", 2),

    # ── ABRAHAM ──
    ("Gen.12.1", 2, 124, "abraham_tested", 3),
    ("Gen.12.1", 6, 74, "abraham_idols", 3),
    ("Gen.15.5", 2, 260, "abraham_faith", 3),
    ("Gen.16.1", 14, 37, "hagar_ishmael", 3),      # Hagar in the valley
    ("Gen.17.1", 2, 127, "kaaba_building", 2),
    ("Gen.18.1", 11, 69, "abraham_guests", 3),     # angels visit
    ("Gen.18.1", 51, 24, "abraham_guests", 3),
    ("Gen.21.1", 37, 101, "sacrifice_son", 3),     # promise of son
    ("Gen.22.1", 37, 102, "sacrifice_son", 3),     # binding/sacrifice
    ("Gen.22.1", 37, 107, "ransom_sacrifice", 3),

    # ── JOSEPH ──
    ("Gen.37.5", 12, 4, "joseph_dream", 3),
    ("Gen.37.18", 12, 8, "brothers_plot", 3),
    ("Gen.37.28", 12, 19, "joseph_sold", 3),
    ("Gen.39.7", 12, 23, "potiphar_wife", 3),
    ("Gen.41.1", 12, 43, "pharaoh_dream", 3),
    ("Gen.45.1", 12, 90, "joseph_reveals", 3),
    ("Gen.50.20", 12, 100, "joseph_reunited", 3),

    # ── MOSES ──
    ("Exod.1.22", 28, 4, "pharaoh_oppression", 3),
    ("Exod.2.1", 20, 38, "moses_basket", 3),
    ("Exod.2.1", 28, 7, "moses_basket", 3),
    ("Exod.3.1", 20, 9, "burning_bush", 3),
    ("Exod.3.1", 28, 29, "burning_bush", 3),
    ("Exod.3.1", 27, 7, "burning_bush", 3),
    ("Exod.7.10", 7, 107, "moses_staff_serpent", 3),
    ("Exod.7.10", 26, 32, "moses_staff_serpent", 3),
    ("Exod.7.20", 7, 133, "plagues", 3),
    ("Exod.14.21", 26, 63, "parting_sea", 3),
    ("Exod.14.21", 44, 24, "parting_sea", 3),
    ("Exod.14.28", 2, 50, "pharaoh_drowns", 3),
    ("Exod.14.28", 10, 90, "pharaoh_drowns", 3),
    ("Exod.20.1", 2, 83, "commandments", 2),
    ("Exod.20.3", 17, 22, "no_other_gods", 3),     # tawhid
    ("Exod.24.12", 7, 145, "tablets", 3),
    ("Exod.32.1", 7, 148, "golden_calf", 3),
    ("Exod.32.1", 20, 85, "golden_calf", 3),
    ("Exod.16.4", 2, 57, "manna_quail", 3),
    ("Exod.16.4", 20, 80, "manna_quail", 3),

    # ── DAVID AND SOLOMON ──
    ("1Sam.17.1", 2, 251, "david_goliath", 3),
    ("2Sam.12.1", 38, 21, "david_parable", 3),
    ("1Kgs.3.16", 27, 15, "solomon_wisdom", 2),
    ("1Kgs.10.1", 27, 22, "queen_sheba", 3),
    ("Ps.21.1", 21, 105, "psalms_zabur", 2),       # Zabur = Psalms
    ("Ps.37.29", 21, 105, "righteous_inherit", 3),

    # ── PROPHETIC BOOKS ──
    ("Isa.7.14", 3, 45, "virgin_birth_prophecy", 2),
    ("Isa.7.14", 19, 20, "virgin_birth", 2),
    ("Isa.9.6", 19, 30, "child_prophecy", 2),
    ("Isa.42.1", 61, 6, "coming_prophet", 2),       # "Ahmad" reference
    ("Jer.31.33", 5, 44, "law_in_hearts", 2),
    ("Ezek.37.1", 2, 259, "valley_dry_bones", 2),   # resurrection parallel
    ("Dan.7.13", 43, 61, "end_times", 2),
    ("Jonah.1.1", 37, 139, "jonah_whale", 3),
    ("Jonah.1.17", 37, 142, "jonah_whale", 3),
    ("Jonah.1.17", 21, 87, "jonah_whale", 3),

    # ── MARY AND JESUS IN QURAN ──
    ("Luke.1.26", 3, 42, "mary_chosen", 3),         # Annunciation
    ("Luke.1.26", 19, 17, "annunciation", 3),
    ("Luke.1.30", 3, 45, "jesus_announced", 3),
    ("Luke.1.34", 19, 20, "virgin_conception", 3),
    ("Luke.1.35", 3, 47, "spirit_conceive", 3),
    ("Luke.2.1", 19, 22, "birth_jesus", 3),
    ("Luke.2.1", 19, 25, "palm_tree", 2),           # birth narrative details
    ("Matt.3.16", 2, 87, "holy_spirit", 2),
    ("Matt.4.23", 3, 49, "jesus_miracles", 3),      # healing, raising dead
    ("Matt.4.23", 5, 110, "jesus_miracles", 3),
    ("John.1.1", 4, 171, "word_of_god", 3),         # Jesus as Word/kalima
    ("John.1.1", 3, 39, "word_of_god", 3),
    ("Matt.26.36", 4, 157, "crucifixion_question", 3),  # crucifixion debate
    ("Matt.28.6", 3, 55, "raised_to_god", 2),       # resurrection/raising
    ("Mark.6.3", 5, 112, "table_from_heaven", 2),   # disciples ask for table
    ("John.14.16", 61, 6, "comforter_ahmad", 2),    # Paraclete = Ahmad?

    # ── SHARED ETHICAL TEACHINGS ──
    ("Exod.20.13", 5, 32, "do_not_kill", 3),
    ("Exod.20.15", 5, 38, "do_not_steal", 2),
    ("Lev.19.18", 2, 195, "love_neighbor", 2),
    ("Deut.6.4", 112, 1, "god_is_one", 3),          # Shema / Al-Ikhlas
    ("Deut.6.4", 2, 163, "god_is_one", 3),
    ("Prov.11.1", 83, 1, "honest_measure", 3),      # fair weights/measures
    ("Isa.1.17", 4, 36, "care_orphans", 2),
    ("Jas.2.15", 2, 177, "charity", 2),

    # ── ESCHATOLOGY / END TIMES ──
    ("Matt.24.29", 81, 1, "end_times_signs", 3),    # cosmic signs
    ("Matt.24.29", 82, 1, "end_times_signs", 3),
    ("Rev.6.12", 99, 1, "earthquake", 2),
    ("Rev.20.12", 39, 69, "book_of_deeds", 3),      # judgment books opened
    ("Rev.20.12", 18, 49, "book_of_deeds", 3),
    ("Matt.25.31", 23, 101, "trumpet_judgment", 2),
    ("1Cor.15.52", 39, 68, "trumpet_blast", 3),
    ("Rev.21.1", 14, 48, "new_heaven_earth", 2),
    ("Rev.22.1", 47, 15, "rivers_paradise", 2),

    # ── ADDITIONAL SHARED FIGURES ──
    ("Gen.19.1", 11, 77, "lot_sodom", 3),           # Lot/Lut
    ("Gen.19.1", 26, 160, "lot_sodom", 3),
    ("Gen.19.26", 7, 83, "lots_wife", 3),
    ("Gen.25.25", 19, 54, "ishmael", 2),             # Ismail
    ("Num.22.21", 2, 248, "ark_of_covenant", 2),     # Tabut/Ark
    ("1Kgs.18.1", 37, 123, "elijah_ilyas", 3),       # Elijah/Ilyas
    ("2Kgs.5.1", 10, 98, "people_of_yunus", 2),
    ("Job.1.1", 21, 83, "job_ayyub", 3),             # Job/Ayyub
    ("Job.1.1", 38, 41, "job_ayyub", 3),

    # ── THEOLOGICAL CONCEPTS ──
    ("Gen.1.3", 36, 82, "divine_command_be", 3),     # "Let there be" / "kun fayakun"
    ("Gen.1.3", 2, 117, "divine_command_be", 3),
    ("Ps.104.1", 55, 1, "divine_attributes", 2),     # ar-Rahman
    ("Isa.40.28", 2, 255, "throne_verse", 2),         # God's majesty
    ("Isa.45.5", 28, 70, "no_god_but_god", 3),
    ("Deut.18.15", 7, 157, "prophet_like_moses", 3),  # prophet prophecy
    ("John.16.13", 16, 89, "spirit_of_truth", 2),
    ("Ps.119.105", 5, 15, "light_guidance", 2),       # scripture as light
    ("Matt.5.3", 57, 23, "humility", 2),
    ("Prov.3.5", 3, 159, "trust_in_god", 2),
]


def parse_verse_ref(ref):
    """Parse 'Gen.1.1' or 'Gen.1.1-Gen.1.3' into (book_idx, chapter, verse)
    Returns the first verse's position."""
    # Handle ranges: take the first reference
    if '-' in ref:
        ref = ref.split('-')[0]

    parts = ref.split('.')
    if len(parts) < 3:
        return None

    book_abbr = parts[0]
    try:
        chapter = int(parts[1])
        verse = int(parts[2])
    except ValueError:
        return None

    if book_abbr not in ABBR_TO_IDX:
        return None

    return (ABBR_TO_IDX[book_abbr], chapter, verse)


def verse_to_position(book_idx, chapter, verse):
    """Convert a verse reference to a linear position (0 to TOTAL_BIBLE_VERSES).
    This is an approximation — we distribute verses evenly across chapters."""
    pos = 0
    for i in range(book_idx):
        pos += BIBLE_BOOKS[i][4]

    # Approximate position within book
    book = BIBLE_BOOKS[book_idx]
    total_chapters = book[3]
    total_verses = book[4]
    # Rough: distribute verses evenly per chapter
    verses_per_chapter = total_verses / total_chapters
    pos += (chapter - 1) * verses_per_chapter + verse

    return pos


def quran_verse_to_position(surah_idx, ayah):
    """Convert a Quran reference to linear position."""
    pos = 0
    for i in range(surah_idx):
        pos += QURAN_SURAHS[i][2]
    pos += ayah
    return pos


def process_bible_crossrefs(filepath, min_votes=10, sample_rate=1.0):
    """Process the OpenBible cross-reference data."""
    refs = []
    count = 0
    skipped = 0

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('From'):
                continue

            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            from_ref, to_ref, votes = parts[0], parts[1], int(parts[2])

            # Filter by vote threshold
            if votes < min_votes:
                skipped += 1
                continue

            from_parsed = parse_verse_ref(from_ref)
            to_parsed = parse_verse_ref(to_ref)

            if from_parsed is None or to_parsed is None:
                skipped += 1
                continue

            from_pos = verse_to_position(*from_parsed)
            to_pos = verse_to_position(*to_parsed)

            from_book = from_parsed[0]
            to_book = to_parsed[0]

            # Determine connection type
            from_testament = BIBLE_BOOKS[from_book][2]
            to_testament = BIBLE_BOOKS[to_book][2]

            if from_testament == "OT" and to_testament == "OT":
                conn_type = 0  # ot_ot
            elif from_testament == "NT" and to_testament == "NT":
                conn_type = 1  # nt_nt
            else:
                conn_type = 2  # ot_nt

            # Compact: [from, to, type, votes]
            refs.append([round(from_pos), round(to_pos), conn_type, votes])
            count += 1

    print(f"Processed {count} cross-references (skipped {skipped})")
    return refs


def process_quran_connections():
    """Process Bible-Quran connections."""
    refs = []

    for bible_ref, surah, ayah, theme, strength in BIBLE_QURAN_CONNECTIONS:
        parsed = parse_verse_ref(bible_ref)
        if parsed is None:
            print(f"Warning: could not parse {bible_ref}")
            continue

        bible_pos = verse_to_position(*parsed)
        quran_pos = quran_verse_to_position(surah - 1, ayah)

        refs.append({
            "biblePos": round(bible_pos, 1),
            "bibleBook": parsed[0],
            "quranPos": round(quran_pos, 1),
            "quranSurah": surah,
            "theme": theme,
            "strength": strength,
        })

    return refs


def build_book_positions():
    """Build the positional metadata for each Bible book."""
    books = []
    pos = 0
    for i, (abbr, name, testament, chapters, verses) in enumerate(BIBLE_BOOKS):
        books.append({
            "abbr": abbr,
            "name": name,
            "testament": testament,
            "start": round(pos, 1),
            "end": round(pos + verses, 1),
            "verses": verses,
        })
        pos += verses
    return books


def build_surah_positions():
    """Build positional metadata for Quran surahs."""
    surahs = []
    pos = 0
    for num, name, verses in QURAN_SURAHS:
        surahs.append({
            "num": num,
            "name": name,
            "start": round(pos, 1),
            "end": round(pos + verses, 1),
            "verses": verses,
        })
        pos += verses
    return surahs


def main():
    print("Processing Bible cross-references...")
    bible_refs = process_bible_crossrefs("cross_references.txt", min_votes=0)

    print("Processing Bible-Quran connections...")
    quran_refs = process_quran_connections()

    print("Building metadata...")
    books = build_book_positions()
    surahs = build_surah_positions()

    data = {
        "totalBibleVerses": TOTAL_BIBLE_VERSES,
        "totalQuranVerses": TOTAL_QURAN_VERSES,
        "books": books,
        "surahs": surahs,
        "bibleRefs": bible_refs,
        "quranRefs": quran_refs,
    }

    outpath = "scripture_data.json"
    with open(outpath, 'w') as f:
        json.dump(data, f)

    print(f"Output: {outpath}")
    print(f"  Bible cross-refs: {len(bible_refs)}")
    print(f"  Bible-Quran connections: {len(quran_refs)}")
    print(f"  Total Bible verses: {TOTAL_BIBLE_VERSES}")
    print(f"  Total Quran verses: {TOTAL_QURAN_VERSES}")
    print(f"  File size: {len(json.dumps(data)) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
