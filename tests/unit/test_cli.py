import pytest
import argparse
from unittest.mock import patch, MagicMock
from memorymesh.cli import cmd_sessions, cmd_stats, main, _get_db_paths, _open_db
import sqlite3
import sys

def test_get_db_paths():
    with patch('memorymesh.config.AppConfig.from_env') as mock_from_env:
        mock_config = MagicMock()
        mock_config.session.db_path = "session.db"
        mock_config.sqlite_vec.db_path = "vec.db"
        mock_config.instinct.db_path = "instinct.db"
        mock_from_env.return_value = mock_config
        
        assert _get_db_paths() == ("session.db", "vec.db", "instinct.db")

def test_open_db():
    with patch('sqlite3.connect') as mock_connect:
        conn = _open_db("test.db")
        mock_connect.assert_called_once_with("test.db")
        assert conn == mock_connect.return_value

@patch('memorymesh.cli._get_db_paths')
@patch('memorymesh.cli._open_db')
@patch('rich.console.Console.print')
def test_cmd_sessions_success(mock_print, mock_open_db, mock_get_db_paths):
    mock_get_db_paths.return_value = ("session.db", "vec.db", "instinct.db")
    
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        {"session_id": "123456789", "user_id": "u1", "status": "active", "workspace_path": "/test", "created_at": "now", "updated_at": "now", "ended_at": None}
    ]
    mock_conn.execute.return_value = mock_cursor
    mock_open_db.return_value = mock_conn
    
    args = argparse.Namespace(limit=10)
    cmd_sessions(args)
    
    mock_open_db.assert_called_once_with("session.db")
    mock_conn.execute.assert_called_once()
    mock_conn.close.assert_called_once()
    mock_print.assert_called_once() # Should print table

@patch('memorymesh.cli._get_db_paths')
@patch('memorymesh.cli._open_db')
@patch('rich.console.Console.print')
def test_cmd_sessions_empty(mock_print, mock_open_db, mock_get_db_paths):
    mock_get_db_paths.return_value = ("session.db", "vec.db", "instinct.db")
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.execute.return_value = mock_cursor
    mock_open_db.return_value = mock_conn
    
    args = argparse.Namespace(limit=10)
    cmd_sessions(args)
    
    mock_print.assert_called_once_with("[yellow]No sessions found.[/yellow]")

@patch('memorymesh.cli._get_db_paths')
@patch('memorymesh.cli._open_db')
def test_cmd_sessions_error(mock_open_db, mock_get_db_paths, capsys):
    mock_get_db_paths.return_value = ("session.db", "vec.db", "instinct.db")
    mock_open_db.side_effect = sqlite3.OperationalError("DB error")
    
    args = argparse.Namespace(limit=10)
    with pytest.raises(SystemExit) as e:
        cmd_sessions(args)
    assert e.value.code == 1
    assert "Error: DB error" in capsys.readouterr().out

@patch('memorymesh.cli._get_db_paths')
@patch('memorymesh.cli._open_db')
@patch('rich.console.Console.print')
@patch('os.path.getsize')
def test_cmd_stats_success(mock_getsize, mock_print, mock_open_db, mock_get_db_paths):
    mock_get_db_paths.return_value = ("session.db", "vec.db", "instinct.db")
    
    def mock_db_behavior(db_path):
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = {"cnt": 5}
        conn.execute.return_value = cursor
        return conn
        
    mock_open_db.side_effect = mock_db_behavior
    mock_getsize.return_value = 2048 # 2KB
    
    args = argparse.Namespace()
    cmd_stats(args)
    
    assert mock_open_db.call_count == 3
    assert mock_print.call_count == 1 # prints panel

@patch('memorymesh.cli._get_db_paths')
@patch('memorymesh.cli._open_db')
@patch('rich.console.Console.print')
def test_cmd_stats_error(mock_print, mock_open_db, mock_get_db_paths):
    mock_get_db_paths.return_value = ("session.db", "vec.db", "instinct.db")
    mock_open_db.side_effect = FileNotFoundError()
    
    args = argparse.Namespace()
    cmd_stats(args)
    
    # Still prints panel, but with all 0s
    mock_print.assert_called_once()

@patch('argparse.ArgumentParser.parse_args')
@patch('memorymesh.cli.cmd_sessions')
@patch('memorymesh.cli.cmd_stats')
@patch('memorymesh.main.cmd_init')
def test_main_routing(mock_cmd_init, mock_cmd_stats, mock_cmd_sessions, mock_parse_args):
    # Test sessions
    mock_parse_args.return_value = argparse.Namespace(command="sessions", limit=10)
    main()
    mock_cmd_sessions.assert_called_once()
    
    # Test stats
    mock_parse_args.return_value = argparse.Namespace(command="stats")
    main()
    mock_cmd_stats.assert_called_once()
    
    # Test init
    mock_parse_args.return_value = argparse.Namespace(command="init", path="somepath")
    main()
    mock_cmd_init.assert_called_once_with(["somepath"])
    
    # Test none
    mock_parse_args.return_value = argparse.Namespace(command=None)
    with patch('argparse.ArgumentParser.print_help') as mock_help:
        main()
        mock_help.assert_called_once()