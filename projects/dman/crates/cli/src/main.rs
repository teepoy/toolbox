use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use colored::Colorize;
use comfy_table::{Cell, Table};
use dman_core::catalog::Catalog;
use dman_core::dataset::DatasetService;
use dman_core::types::DatasetFormat;

#[derive(Parser)]
#[command(
    name = "dman",
    about = "Dataset Manager — manage your ML datasets",
    version,
    propagate_version = true
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize the dman catalog (~/.dman or $DMAN_HOME)
    Init,

    /// Register a dataset with the catalog
    Add {
        /// Dataset name (must be unique)
        name: String,
        /// Path to the dataset directory or file
        path: PathBuf,
        /// Dataset format
        #[arg(long, value_enum, default_value = "custom")]
        format: FormatArg,
    },

    /// List all registered datasets
    List {
        /// Output format
        #[arg(long, value_enum, default_value = "table")]
        format: ListFormat,
    },

    /// Show detailed information about a dataset
    Inspect {
        /// Dataset name
        name: String,
    },

    /// Remove a dataset from the catalog (does not delete files)
    Remove {
        /// Dataset name
        name: String,
        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Import a dataset from a path (not yet implemented)
    Import {
        /// Path to import from
        path: PathBuf,
        /// Dataset format
        #[arg(long, value_enum)]
        format: Option<FormatArg>,
        /// Name to assign to the imported dataset
        #[arg(long)]
        name: Option<String>,
    },

    /// Export a dataset to a path (not yet implemented)
    Export {
        /// Dataset name
        name: String,
        /// Output path
        output: PathBuf,
        /// Target format
        #[arg(long, value_enum)]
        format: FormatArg,
    },

    /// Apply operations to a dataset (not yet implemented)
    Operate,

    /// Manage virtual datasets (not yet implemented)
    Virtual,

    /// Start the dman HTTP server (not yet implemented)
    Serve {
        /// Port to listen on
        #[arg(long, short, default_value = "8080")]
        port: u16,
    },

    /// Launch the terminal UI (not yet implemented)
    Tui,

    /// Materialize a virtual dataset (not yet implemented)
    Materialize {
        /// Dataset name
        name: String,
    },
}

#[derive(Clone, Debug, ValueEnum)]
enum FormatArg {
    Yolo,
    Coco,
    Hf,
    Custom,
}

impl From<FormatArg> for DatasetFormat {
    fn from(arg: FormatArg) -> Self {
        match arg {
            FormatArg::Yolo => DatasetFormat::Yolo,
            FormatArg::Coco => DatasetFormat::Coco,
            FormatArg::Hf => DatasetFormat::HuggingFace,
            FormatArg::Custom => DatasetFormat::Custom("custom".to_string()),
        }
    }
}

#[derive(Clone, ValueEnum)]
enum ListFormat {
    Table,
    Json,
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("{} {}", "error:".red().bold(), e);
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Init => cmd_init(),
        Commands::Add { name, path, format } => cmd_add(&name, &path, format.into()),
        Commands::List { format } => cmd_list(format),
        Commands::Inspect { name } => cmd_inspect(&name),
        Commands::Remove { name, yes } => cmd_remove(&name, yes),
        Commands::Import { path, format, name } => cmd_import_stub(&path, format, name.as_deref()),
        Commands::Export {
            name,
            output,
            format,
        } => cmd_export_stub(&name, &output, format),
        Commands::Operate => cmd_stub("operate"),
        Commands::Virtual => cmd_stub("virtual"),
        Commands::Serve { port } => cmd_serve_stub(port),
        Commands::Tui => cmd_stub("tui"),
        Commands::Materialize { name } => cmd_materialize_stub(&name),
    }
}

fn cmd_init() -> Result<()> {
    let home = Catalog::home_path();
    Catalog::init()
        .with_context(|| format!("failed to initialize catalog at {}", home.display()))?;
    println!(
        "{} Catalog initialized at {}",
        "✓".green().bold(),
        home.display().to_string().cyan()
    );
    Ok(())
}

fn cmd_add(name: &str, path: &PathBuf, format: DatasetFormat) -> Result<()> {
    let catalog = open_catalog()?;
    let ds = DatasetService::register(catalog.db(), name, path, format)
        .with_context(|| format!("failed to register dataset '{name}'"))?;
    println!(
        "{} Registered dataset '{}' (id={}, format={:?})",
        "✓".green().bold(),
        ds.name.cyan(),
        ds.id,
        ds.format
    );
    Ok(())
}

fn cmd_list(format: ListFormat) -> Result<()> {
    let catalog = open_catalog()?;
    let datasets = DatasetService::list(catalog.db()).context("failed to list datasets")?;

    match format {
        ListFormat::Json => {
            let json = serde_json::to_string_pretty(&datasets)
                .context("failed to serialize datasets to JSON")?;
            println!("{json}");
        }
        ListFormat::Table => {
            if datasets.is_empty() {
                println!(
                    "{}",
                    "No datasets registered. Use `dman add` to register one.".yellow()
                );
                return Ok(());
            }
            let mut table = Table::new();
            table.set_header(vec![
                Cell::new("ID"),
                Cell::new("Name"),
                Cell::new("Format"),
                Cell::new("Path"),
                Cell::new("Created"),
            ]);
            for ds in &datasets {
                let format_str = match &ds.format {
                    DatasetFormat::Yolo => "yolo".to_string(),
                    DatasetFormat::Coco => "coco".to_string(),
                    DatasetFormat::HuggingFace => "hf".to_string(),
                    DatasetFormat::Custom(s) => format!("custom:{s}"),
                };
                table.add_row(vec![
                    Cell::new(ds.id),
                    Cell::new(&ds.name),
                    Cell::new(format_str),
                    Cell::new(ds.path.display()),
                    Cell::new(&ds.created_at),
                ]);
            }
            println!("{table}");
            println!("{} dataset(s) total.", datasets.len().to_string().bold());
        }
    }
    Ok(())
}

fn cmd_inspect(name: &str) -> Result<()> {
    let catalog = open_catalog()?;
    let info = DatasetService::inspect(catalog.db(), name)
        .with_context(|| format!("failed to inspect dataset '{name}'"))?;

    let ds = &info.dataset;
    let format_str = match &ds.format {
        DatasetFormat::Yolo => "yolo".to_string(),
        DatasetFormat::Coco => "coco".to_string(),
        DatasetFormat::HuggingFace => "hf".to_string(),
        DatasetFormat::Custom(s) => format!("custom:{s}"),
    };

    println!("{}", "─".repeat(50).dimmed());
    println!("  {} {}", "Name:".bold(), ds.name.cyan());
    println!("  {} {}", "ID:".bold(), ds.id);
    println!("  {} {}", "Format:".bold(), format_str);
    println!("  {} {}", "Path:".bold(), ds.path.display());
    println!("  {} {}", "Created:".bold(), ds.created_at);
    if let Some(updated) = &ds.updated_at {
        println!("  {} {}", "Updated:".bold(), updated);
    }
    println!("{}", "─".repeat(50).dimmed());
    println!("  {} {}", "Images:".bold(), info.image_count);
    println!("  {} {}", "Annotations:".bold(), info.annotation_count);
    println!("  {} {} B", "Disk size:".bold(), info.disk_size_bytes);

    if !info.categories.is_empty() {
        println!("{}", "─".repeat(50).dimmed());
        println!("  {}:", "Categories".bold());
        for cat in &info.categories {
            let sup = cat
                .supercategory
                .as_deref()
                .map(|s| format!(" ({})", s))
                .unwrap_or_default();
            println!("    • {}{}", cat.name, sup);
        }
    }

    if let Some(meta) = &ds.metadata {
        println!("{}", "─".repeat(50).dimmed());
        println!("  {}:", "Metadata".bold());
        let pretty = serde_json::to_string_pretty(meta).unwrap_or_default();
        for line in pretty.lines() {
            println!("    {line}");
        }
    }

    println!("{}", "─".repeat(50).dimmed());
    Ok(())
}

fn cmd_remove(name: &str, yes: bool) -> Result<()> {
    if !yes {
        print!(
            "Remove dataset '{}'? This cannot be undone. [y/N] ",
            name.yellow()
        );
        io::stdout().flush().context("failed to flush stdout")?;
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .context("failed to read confirmation")?;
        if !matches!(input.trim().to_lowercase().as_str(), "y" | "yes") {
            println!("{}", "Aborted.".dimmed());
            return Ok(());
        }
    }
    let catalog = open_catalog()?;
    DatasetService::remove(catalog.db(), name)
        .with_context(|| format!("failed to remove dataset '{name}'"))?;
    println!("{} Removed dataset '{}'.", "✓".green().bold(), name.cyan());
    Ok(())
}

fn cmd_import_stub(path: &PathBuf, format: Option<FormatArg>, name: Option<&str>) -> Result<()> {
    eprintln!(
        "{} import is not yet implemented (path={}, format={:?}, name={:?})",
        "stub:".yellow().bold(),
        path.display(),
        format.map(|f| format!("{f:?}")),
        name
    );
    Ok(())
}

fn cmd_export_stub(name: &str, output: &PathBuf, _format: FormatArg) -> Result<()> {
    eprintln!(
        "{} export is not yet implemented (name={name}, output={})",
        "stub:".yellow().bold(),
        output.display()
    );
    Ok(())
}

fn cmd_serve_stub(port: u16) -> Result<()> {
    eprintln!(
        "{} serve is not yet implemented (port={port})",
        "stub:".yellow().bold()
    );
    Ok(())
}

fn cmd_materialize_stub(name: &str) -> Result<()> {
    eprintln!(
        "{} materialize is not yet implemented (name={name})",
        "stub:".yellow().bold()
    );
    Ok(())
}

fn cmd_stub(cmd: &str) -> Result<()> {
    eprintln!("{} `{cmd}` is not yet implemented", "stub:".yellow().bold());
    Ok(())
}

fn open_catalog() -> Result<Catalog> {
    Catalog::open().context("catalog not found — run `dman init` first")
}
