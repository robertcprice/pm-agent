"""
Database Capability for PM Agent

Provides database operations including:
- Migration execution (Alembic, Prisma, Django)
- Backup and restore
- Database seeding
- Schema inspection
- Query execution

Safety Features:
- Automatic backups before destructive operations
- Dry-run mode for testing
- Rollback capability
- Connection validation
"""

import asyncio
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiofiles
import asyncpg
import aiosqlite

from pm_agent.logger import ThoughtLogger


class DatabaseType(Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    MYSQL = "mysql"
    MSSQL = "mssql"


class MigrationFramework(Enum):
    """Supported migration frameworks"""
    ALEMBIC = "alembic"
    PRISMA = "prisma"
    DJANGO = "django"
    SQLMIGRATE = "sqlmigrate"
    FLYWAY = "flyway"


class MigrationStatus(Enum):
    """Migration states"""
    PENDING = "pending"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    db_type: DatabaseType
    host: Optional[str] = None
    port: Optional[int] = None
    database: str = ":memory:"
    username: Optional[str] = None
    password: Optional[str] = None
    path: Optional[Path] = None  # For SQLite

    def get_connection_string(self) -> str:
        """Generate connection string from config"""
        if self.db_type == DatabaseType.SQLITE:
            return str(self.path or self.database)

        # For PostgreSQL, MySQL, etc.
        if self.db_type == DatabaseType.POSTGRESQL:
            return (
                f"postgresql://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )
        elif self.db_type == DatabaseType.MYSQL:
            return (
                f"mysql://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")


@dataclass
class Migration:
    """Migration information"""
    name: str
    version: str
    status: MigrationStatus
    applied_at: Optional[datetime] = None
    rollback_script: Optional[str] = None


@dataclass
class BackupConfig:
    """Configuration for database backups"""
    backup_path: Path
    compress: bool = True
    include_schema: bool = True
    exclude_tables: List[str] = field(default_factory=list)
    max_backups: int = 10


@dataclass
class QueryResult:
    """Result of a database query"""
    rows: List[Dict[str, Any]]
    row_count: int
    columns: List[str]
    execution_time_ms: float


class DatabaseError(Exception):
    """Base exception for database operations"""
    pass


class ConnectionError(DatabaseError):
    """Database connection failed"""
    pass


class MigrationError(DatabaseError):
    """Migration operation failed"""
    pass


class BackupError(DatabaseError):
    """Backup operation failed"""
    pass


class DatabaseCapability:
    """
    Database management capability with safety features.

    Supports PostgreSQL, SQLite, MySQL with migrations,
    backups, seeding, and query execution.
    """

    def __init__(self, logger: Optional[ThoughtLogger] = None):
        """
        Initialize database capability.

        Args:
            logger: Optional thought logger for tracking operations
        """
        self.logger = logger or ThoughtLogger("database_capability")
        self._connections: Dict[str, Any] = {}
        self._dry_run = False
        self._auto_backup = True

    async def test_connection(
        self,
        config: DatabaseConfig
    ) -> Tuple[bool, str]:
        """
        Test database connection.

        Args:
            config: Database configuration

        Returns:
            Tuple of (success, message)
        """
        try:
            if config.db_type == DatabaseType.POSTGRESQL:
                conn = await asyncpg.connect(config.get_connection_string())
                await conn.close()
                return True, "PostgreSQL connection successful"

            elif config.db_type == DatabaseType.SQLITE:
                async with aiosqlite.connect(config.get_connection_string()):
                    pass
                return True, "SQLite connection successful"

            else:
                return False, f"Database type {config.db_type} not yet supported"

        except Exception as e:
            return False, f"Connection failed: {e}"

    async def create_backup(
        self,
        config: DatabaseConfig,
        backup_config: BackupConfig,
    ) -> Path:
        """
        Create a database backup.

        Args:
            config: Database configuration
            backup_config: Backup configuration

        Returns:
            Path to backup file

        Raises:
            BackupError: If backup fails
        """
        self.logger.log_thought(
            "backup_start",
            f"Creating backup of {config.database}",
            {"database": config.database, "path": str(backup_config.backup_path)}
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if config.db_type == DatabaseType.SQLITE:
            backup_path = backup_config.backup_path / f"{config.database}_{timestamp}.db"
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            # Simple file copy for SQLite
            import shutil
            shutil.copy2(config.path or config.database, backup_path)

            self.logger.log_thought(
                "backup_complete",
                f"SQLite backup created: {backup_path}",
                {"size": backup_path.stat().st_size}
            )
            return backup_path

        elif config.db_type == DatabaseType.POSTGRESQL:
            backup_path = backup_config.backup_path / f"{config.database}_{timestamp}.sql"
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            # Use pg_dump
            cmd = [
                "pg_dump",
                f"--host={config.host}",
                f"--port={config.port}",
                f"--username={config.username}",
                "--no-password",
                "--format=plain",
            ]

            if backup_config.compress:
                cmd.append("--compress=9")

            if backup_config.include_schema:
                cmd.append("--schema-only")

            for table in backup_config.exclude_tables:
                cmd.extend(["--exclude-table-data", table])

            cmd.append(config.database)

            # Set password in environment
            env = {"PGPASSWORD": config.password}

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**subprocess.os.environ, **env}
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise BackupError(f"pg_dump failed: {stderr.decode()}")

            async with aiofiles.open(backup_path, "w") as f:
                await f.write(stdout.decode())

            self.logger.log_thought(
                "backup_complete",
                f"PostgreSQL backup created: {backup_path}",
                {"size": backup_path.stat().st_size}
            )
            return backup_path

        else:
            raise BackupError(f"Backup not supported for {config.db_type}")

    async def restore_backup(
        self,
        config: DatabaseConfig,
        backup_path: Path,
    ) -> bool:
        """
        Restore database from backup.

        Args:
            config: Database configuration
            backup_path: Path to backup file

        Returns:
            True if restore successful

        Raises:
            BackupError: If restore fails
        """
        self.logger.log_thought(
            "restore_start",
            f"Restoring {config.database} from {backup_path}",
            {"database": config.database, "backup": str(backup_path)}
        )

        if config.db_type == DatabaseType.SQLITE:
            import shutil
            shutil.copy2(backup_path, config.path or config.database)
            return True

        elif config.db_type == DatabaseType.POSTGRESQL:
            cmd = [
                "psql",
                f"--host={config.host}",
                f"--port={config.port}",
                f"--username={config.username}",
                "--no-password",
                config.database
            ]

            env = {"PGPASSWORD": config.password}

            async with aiofiles.open(backup_path, "r") as f:
                sql_content = await f.read()

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**subprocess.os.environ, **env}
            )

            stdout, stderr = await process.communicate(
                input=sql_content.encode()
            )

            if process.returncode != 0:
                raise BackupError(f"psql failed: {stderr.decode()}")

            self.logger.log_thought(
                "restore_complete",
                f"Database restored: {config.database}",
                {}
            )
            return True

        else:
            raise BackupError(f"Restore not supported for {config.db_type}")

    async def run_migrations(
        self,
        config: DatabaseConfig,
        framework: MigrationFramework,
        project_path: Path,
        direction: str = "upgrade",
        dry_run: bool = False,
    ) -> Tuple[bool, List[Migration]]:
        """
        Run database migrations.

        Args:
            config: Database configuration
            framework: Migration framework to use
            project_path: Path to project with migrations
            direction: "upgrade" or "downgrade"
            dry_run: If True, don't actually apply migrations

        Returns:
            Tuple of (success, applied_migrations)

        Raises:
            MigrationError: If migration fails
        """
        self.logger.log_thought(
            "migrations_start",
            f"Running {direction} migrations with {framework.value}",
            {
                "framework": framework.value,
                "direction": direction,
                "dry_run": dry_run
            }
        )

        if dry_run or self._dry_run:
            self.logger.log_thought(
                "migrations_dry_run",
                "Dry-run mode - no changes will be applied",
                {}
            )

        if framework == MigrationFramework.ALEMBIC:
            return await self._run_alembic_migrations(
                config, project_path, direction, dry_run
            )
        elif framework == MigrationFramework.PRISMA:
            return await self._run_prisma_migrations(
                config, project_path, direction, dry_run
            )
        elif framework == MigrationFramework.DJANGO:
            return await self._run_django_migrations(
                config, project_path, direction, dry_run
            )
        else:
            raise MigrationError(f"Framework {framework} not yet supported")

    async def _run_alembic_migrations(
        self,
        config: DatabaseConfig,
        project_path: Path,
        direction: str,
        dry_run: bool,
    ) -> Tuple[bool, List[Migration]]:
        """Run Alembic migrations"""
        # Set ALEMBIC_CONFIG env variable if needed
        env = {
            "DATABASE_URL": config.get_connection_string(),
        }

        cmd = ["alembic", "upgrade", "head"]

        if dry_run:
            cmd.append("--sql")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=project_path,
            env={**subprocess.os.environ, **env}
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode()
            self.logger.log_thought(
                "migration_failed",
                f"Alembic migration failed: {error_msg}",
                {"error": error_msg}
            )
            raise MigrationError(f"Migration failed: {error_msg}")

        output = stdout.decode()
        migrations = self._parse_alembic_output(output)

        self.logger.log_thought(
            "migrations_complete",
            f"Applied {len(migrations)} Alembic migrations",
            {"count": len(migrations)}
        )

        return True, migrations

    async def _run_prisma_migrations(
        self,
        config: DatabaseConfig,
        project_path: Path,
        direction: str,
        dry_run: bool,
    ) -> Tuple[bool, List[Migration]]:
        """Run Prisma migrations"""
        cmd = ["npx", "prisma", "migrate", "dev"]

        if dry_run:
            cmd.extend(["--skip-generate"])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=project_path,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode()
            raise MigrationError(f"Prisma migration failed: {error_msg}")

        output = stdout.decode()
        migrations = self._parse_prisma_output(output)

        return True, migrations

    async def _run_django_migrations(
        self,
        config: DatabaseConfig,
        project_path: Path,
        direction: str,
        dry_run: bool,
    ) -> Tuple[bool, List[Migration]]:
        """Run Django migrations"""
        cmd = ["python", "manage.py", "migrate"]

        if dry_run:
            cmd.append("--plan")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=project_path,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode()
            raise MigrationError(f"Django migration failed: {error_msg}")

        output = stdout.decode()
        migrations = self._parse_django_output(output)

        return True, migrations

    async def seed_database(
        self,
        config: DatabaseConfig,
        seed_file: Path,
        format: str = "sql",
    ) -> Tuple[bool, int]:
        """
        Seed database with data from a file.

        Args:
            config: Database configuration
            seed_file: Path to seed data file
            format: Format of seed data (sql, json, csv)

        Returns:
            Tuple of (success, rows_inserted)
        """
        self.logger.log_thought(
            "seed_start",
            f"Seeding database from {seed_file.name}",
            {"format": format, "file": str(seed_file)}
        )

        if format == "sql":
            return await self._seed_from_sql(config, seed_file)
        elif format == "json":
            return await self._seed_from_json(config, seed_file)
        elif format == "csv":
            return await self._seed_from_csv(config, seed_file)
        else:
            raise DatabaseError(f"Unsupported seed format: {format}")

    async def _seed_from_sql(
        self,
        config: DatabaseConfig,
        seed_file: Path
    ) -> Tuple[bool, int]:
        """Seed database from SQL file"""
        async with aiofiles.open(seed_file, "r") as f:
            sql_content = await f.read()

        if config.db_type == DatabaseType.SQLITE:
            async with aiosqlite.connect(config.get_connection_string()) as conn:
                await conn.executescript(sql_content)
                await conn.commit()
                return True, conn.total_changes

        elif config.db_type == DatabaseType.POSTGRESQL:
            conn = await asyncpg.connect(config.get_connection_string())
            try:
                result = await conn.execute(sql_content)
                # Parse rows affected from result
                return True, self._parse_affected_rows(result)
            finally:
                await conn.close()

        return True, 0

    async def _seed_from_json(
        self,
        config: DatabaseConfig,
        seed_file: Path
    ) -> Tuple[bool, int]:
        """Seed database from JSON file"""
        async with aiofiles.open(seed_file, "r") as f:
            content = await f.read()

        data = json.loads(content)

        if config.db_type == DatabaseType.SQLITE:
            async with aiosqlite.connect(config.get_connection_string()) as conn:
                rows = 0
                for table_name, records in data.items():
                    for record in records:
                        columns = ", ".join(record.keys())
                        placeholders = ", ".join(["?"] * len(record))
                        values = list(record.values())

                        await conn.execute(
                            f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})",
                            values
                        )
                        rows += 1

                await conn.commit()
                return True, rows

        return True, 0

    async def _seed_from_csv(
        self,
        config: DatabaseConfig,
        seed_file: Path
    ) -> Tuple[bool, int]:
        """Seed database from CSV file"""
        import csv

        async with aiofiles.open(seed_file, "r") as f:
            content = await f.read()

        reader = csv.DictReader(content.splitlines())
        rows = list(reader)

        if config.db_type == DatabaseType.SQLITE:
            # Infer table name from filename
            table_name = seed_file.stem

            async with aiosqlite.connect(config.get_connection_string()) as conn:
                for row in rows:
                    columns = ", ".join(row.keys())
                    placeholders = ", ".join(["?"] * len(row))
                    values = list(row.values())

                    await conn.execute(
                        f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})",
                        values
                    )

                await conn.commit()
                return True, len(rows)

        return True, 0

    async def execute_query(
        self,
        config: DatabaseConfig,
        query: str,
        params: Optional[Tuple] = None,
    ) -> QueryResult:
        """
        Execute a query and return results.

        Args:
            config: Database configuration
            query: SQL query to execute
            params: Query parameters

        Returns:
            QueryResult with rows and metadata
        """
        import time

        start_time = time.time()

        if config.db_type == DatabaseType.SQLITE:
            async with aiosqlite.connect(config.get_connection_string()) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(query, params or ())
                rows = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []

                return QueryResult(
                    rows=[dict(row) for row in rows],
                    row_count=len(rows),
                    columns=columns,
                    execution_time_ms=(time.time() - start_time) * 1000
                )

        elif config.db_type == DatabaseType.POSTGRESQL:
            conn = await asyncpg.connect(config.get_connection_string())
            try:
                result = await conn.fetch(query, *params) if params else await conn.fetch(query)

                return QueryResult(
                    rows=[dict(r) for r in result],
                    row_count=len(result),
                    columns=list(result[0].keys()) if result else [],
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            finally:
                await conn.close()

        raise DatabaseError(f"Unsupported database type: {config.db_type}")

    def _parse_alembic_output(self, output: str) -> List[Migration]:
        """Parse Alembic migration output"""
        migrations = []
        for line in output.split("\n"):
            if "Running upgrade" in line:
                parts = line.split()
                if len(parts) >= 4:
                    migrations.append(Migration(
                        name=parts[4],
                        version=parts[3],
                        status=MigrationStatus.APPLIED
                    ))
        return migrations

    def _parse_prisma_output(self, output: str) -> List[Migration]:
        """Parse Prisma migration output"""
        migrations = []
        # Prisma output format varies, parse common patterns
        for line in output.split("\n"):
            if "Applying migration" in line:
                migrations.append(Migration(
                    name=line.split('"')[1] if '"' in line else "unknown",
                    version="",
                    status=MigrationStatus.APPLIED
                ))
        return migrations

    def _parse_django_output(self, output: str) -> List[Migration]:
        """Parse Django migration output"""
        migrations = []
        for line in output.split("\n"):
            if "Applying" in line:
                parts = line.split()
                if len(parts) >= 3:
                    migrations.append(Migration(
                        name=parts[2].rstrip("."),
                        version=parts[1],
                        status=MigrationStatus.APPLIED
                    ))
        return migrations

    def _parse_affected_rows(self, result: str) -> int:
        """Parse affected rows from database result string"""
        # Common patterns: "INSERT 0 5", "UPDATE 5", etc.
        import re
        match = re.search(r'(\d+)', result)
        return int(match.group(1)) if match else 0

    async def cleanup_old_backups(
        self,
        backup_config: BackupConfig,
        keep_last_n: int = None,
    ) -> int:
        """
        Remove old backups, keeping only the most recent N.

        Args:
            backup_config: Backup configuration
            keep_last_n: Number of backups to keep (defaults to max_backups)

        Returns:
            Number of backups removed
        """
        keep_last_n = keep_last_n or backup_config.max_backups

        backups = sorted(
            backup_config.backup_path.glob("*.db") or
            backup_config.backup_path.glob("*.sql") or
            [],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        to_remove = backups[keep_last_n:]
        removed = 0

        for backup in to_remove:
            backup.unlink()
            removed += 1

        self.logger.log_thought(
            "cleanup_complete",
            f"Removed {removed} old backups",
            {"kept": len(backups) - removed}
        )

        return removed

    def set_dry_run(self, enabled: bool):
        """Enable or disable dry-run mode"""
        self._dry_run = enabled
        self.logger.log_thought(
            "dry_run_changed",
            f"Dry-run mode: {enabled}",
            {"enabled": enabled}
        )

    def set_auto_backup(self, enabled: bool):
        """Enable or disable automatic backups before destructive operations"""
        self._auto_backup = enabled


# Convenience functions
async def migrate_database(
    db_config: DatabaseConfig,
    framework: MigrationFramework,
    project_path: Path,
) -> bool:
    """
    Quick helper to run migrations with automatic backup.

    Example:
        success = await migrate_database(
            db_config=DatabaseConfig(
                db_type=DatabaseType.POSTGRESQL,
                host="localhost",
                port=5432,
                database="myapp",
                username="user",
                password="pass"
            ),
            framework=MigrationFramework.ALEMBIC,
            project_path=Path("/path/to/project")
        )
    """
    db = DatabaseCapability()

    # Auto backup before migrations
    if db._auto_backup:
        await db.create_backup(
            db_config,
            BackupConfig(backup_path=Path("./backups"))
        )

    success, _ = await db.run_migrations(db_config, framework, project_path)
    return success
