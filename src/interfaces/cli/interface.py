"""Main CLI interface"""

import argparse
from typing import Dict, Any

from .commands.pipeline import PipelineCommand
from .commands.status import StatusCommand
from .commands.data import DataCommand
from .formatters import ResultFormatter

class CLIInterface:
    """Command Line Interface for the Trading System"""
    
    def __init__(self):
        """Initialize CLI interface"""
        self.parser = self._create_parser()
        self.formatter = ResultFormatter()
        
        # Initialize command handlers
        self.commands = {
            'pipeline': PipelineCommand(),
            'status': StatusCommand(),
            'data': DataCommand()
        }

    def run(self, args=None) -> int:
        """Run the CLI with given arguments"""
        try:
            parsed_args = self.parser.parse_args(args)
            
            if not parsed_args.command:
                self.parser.print_help()
                return 1
            
            # Route to appropriate command
            command = self.commands.get(parsed_args.command)
            if command:
                return command.execute(parsed_args, self.formatter)
            else:
                print(f"❌ Unknown command: {parsed_args.command}")
                return 1
                
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
            return 1
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            return 1

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser"""
        parser = argparse.ArgumentParser(description='AI Trading System CLI')
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Add command parsers
        for command_name, command in self.commands.items():
            command.add_parser(subparsers)
        
        return parser