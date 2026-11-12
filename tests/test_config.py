"""Tests for brain.config (.env parsing and runtime configuration)."""

from brain.config import Config, load_config, parse_env_file


def _write(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_missing_file_returns_empty_dict(tmp_path):
    assert parse_env_file(tmp_path / "nope.env") == {}


def test_parse_skips_comments_blank_lines_and_lines_without_equals(tmp_path):
    path = _write(tmp_path, "# comment\n\n   \nNOT_A_PAIR\nNAO_IP=10.0.0.5\n")
    assert parse_env_file(path) == {"NAO_IP": "10.0.0.5"}


def test_parse_strips_matching_double_and_single_quotes(tmp_path):
    path = _write(tmp_path, 'A="hello world"\nB=\'spaced\'\n')
    assert parse_env_file(path) == {"A": "hello world", "B": "spaced"}


def test_parse_keeps_unmatched_or_inner_quotes(tmp_path):
    path = _write(tmp_path, "A=\"open\nB='mixed\"\nC=it's\n")
    assert parse_env_file(path) == {"A": '"open', "B": "'mixed\"", "C": "it's"}


def test_parse_trims_whitespace_around_key_and_value(tmp_path):
    path = _write(tmp_path, "  KEY  =   value  \n")
    assert parse_env_file(path) == {"KEY": "value"}


def test_parse_value_may_contain_equals_sign(tmp_path):
    path = _write(tmp_path, "TOKEN=abc=def==\n")
    assert parse_env_file(path) == {"TOKEN": "abc=def=="}


def test_load_config_defaults_when_nothing_set(tmp_path):
    cfg = load_config(env_path=tmp_path / "missing.env", environ={})
    assert cfg == Config()
    assert cfg.nao_ip == "127.0.0.1"
    assert cfg.nao_port == 9559
    assert cfg.music_dir == "songs"
    assert cfg.champions_song == ""
    assert cfg.has_groq is False


def test_load_config_reads_values_from_file(tmp_path):
    path = _write(
        tmp_path,
        "NAO_IP=192.168.1.20\nNAO_PORT=9600\nGROQ_API_KEY=gsk_test\n"
        "MUSIC_DIR=mi_musica\nCHAMPIONS_SONG=champs.mp3\n",
    )
    cfg = load_config(env_path=path, environ={})
    assert cfg.nao_ip == "192.168.1.20"
    assert cfg.nao_port == 9600
    assert cfg.groq_api_key == "gsk_test"
    assert cfg.music_dir == "mi_musica"
    assert cfg.champions_song == "champs.mp3"


def test_environment_overrides_file(tmp_path):
    path = _write(tmp_path, "NAO_IP=10.0.0.1\nNAO_PORT=9600\n")
    cfg = load_config(env_path=path, environ={"NAO_IP": "10.0.0.2", "NAO_PORT": "9700"})
    assert cfg.nao_ip == "10.0.0.2"
    assert cfg.nao_port == 9700


def test_file_value_used_when_environment_lacks_key(tmp_path):
    path = _write(tmp_path, "NAO_IP=10.0.0.1\n")
    cfg = load_config(env_path=path, environ={"NAO_PORT": "9600"})
    assert cfg.nao_ip == "10.0.0.1"
    assert cfg.nao_port == 9600


def test_bad_nao_port_falls_back_to_9559(tmp_path):
    cfg = load_config(env_path=tmp_path / "missing.env", environ={"NAO_PORT": "not-a-port"})
    assert cfg.nao_port == 9559


def test_bad_nao_port_in_file_falls_back_to_9559(tmp_path):
    path = _write(tmp_path, "NAO_PORT=abc\n")
    assert load_config(env_path=path, environ={}).nao_port == 9559


def test_has_groq_true_only_with_non_empty_key(tmp_path):
    with_key = load_config(env_path=tmp_path / "x.env", environ={"GROQ_API_KEY": "gsk_123"})
    empty_key = load_config(env_path=tmp_path / "x.env", environ={"GROQ_API_KEY": ""})
    assert with_key.has_groq is True
    assert empty_key.has_groq is False
