use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use colored::Colorize;
use comfy_table::{Cell, Table};
use dman_core::catalog::Catalog;
use dman_core::dataset::DatasetService;
use dman_core::formats::FormatRegistry;
use dman_core::storage::StorageManager;
use dman_core::types::DatasetFormat;
use dman_server::label_studio::LabelStudioClient;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

mod tui;

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
        #[arg(long)]
        format: String,
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
        #[arg(long)]
        format: Option<String>,
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
        #[arg(long)]
        format: String,
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

    LabelStudio {
        #[command(subcommand)]
        cmd: LsCommand,
    },

    Embed {
        dataset_name: String,
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "32")]
        batch_size: usize,
    },

    /// Launch the interactive terminal UI
    Tui,

    /// Materialize a virtual dataset (not yet implemented)
    Materialize {
        /// Dataset name
        name: String,
    },
}

#[derive(Subcommand)]
enum LsCommand {
    Import {
        url: String,
        api_key: String,
        project: i64,
        name: String,
    },
    Export {
        url: String,
        api_key: String,
        project: i64,
        dataset: String,
        #[arg(long, short, default_value = "8080")]
        port: u16,
    },
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
    let registry = build_registry();
    match cli.command {
        Commands::Init => cmd_init(),
        Commands::Add { name, path, format } => cmd_add(&name, &path, DatasetFormat::new(format)),
        Commands::List { format } => cmd_list(format),
        Commands::Inspect { name } => cmd_inspect(&name),
        Commands::Remove { name, yes } => cmd_remove(&name, yes),
        Commands::Import { path, format, name } => {
            cmd_import(&registry, &path, format.as_deref(), name.as_deref())
        }
        Commands::Export {
            name,
            output,
            format,
        } => cmd_export(&registry, &name, &output, &format),
        Commands::Operate => cmd_stub("operate"),
        Commands::Virtual => cmd_stub("virtual"),
        Commands::Serve { port } => cmd_serve_stub(port),
        Commands::LabelStudio { cmd } => cmd_label_studio(cmd),
        Commands::Embed {
            dataset_name,
            model,
            batch_size,
        } => cmd_embed(&dataset_name, &model, batch_size),
        Commands::Tui => tui::run(),
        Commands::Materialize { name } => cmd_materialize_stub(&name),
    }
}

fn cmd_label_studio(cmd: LsCommand) -> Result<()> {
    match cmd {
        LsCommand::Import {
            url,
            api_key,
            project,
            name,
        } => cmd_label_studio_import(&url, &api_key, project, &name),
        LsCommand::Export {
            url,
            api_key,
            project,
            dataset,
            port,
        } => cmd_label_studio_export(&url, &api_key, project, &dataset, port),
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

fn cmd_add(name: &str, path: &Path, format: DatasetFormat) -> Result<()> {
    let catalog = open_catalog()?;
    let ds = DatasetService::register(catalog.db(), name, path, format)
        .with_context(|| format!("failed to register dataset '{name}'"))?;
    println!(
        "{} Registered dataset '{}' (id={}, format={})",
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
                let format_str = ds.format.to_string();
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
    let format_str = ds.format.to_string();

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
    println!("  {} {}", "Samples:".bold(), info.sample_count);
    println!("  {} {}", "Assets:".bold(), info.asset_count);
    println!("  {} {}", "Annotations:".bold(), info.annotation_count);
    println!("  {} {} B", "Disk size:".bold(), info.disk_size_bytes);

    if info.category_count > 0 {
        println!("{}", "─".repeat(50).dimmed());
        println!("  {} {}", "Categories:".bold(), info.category_count);
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

fn cmd_import(
    registry: &FormatRegistry,
    path: &Path,
    format: Option<&str>,
    name: Option<&str>,
) -> Result<()> {
    let catalog = open_catalog()?;
    let storage = StorageManager::new(catalog.data_path());
    let format_id = match format {
        Some(value) => DatasetFormat::new(value).to_string(),
        None => registry
            .detect_format(path)
            .map(str::to_string)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no registered format provider detected for {}",
                    path.display()
                )
            })?,
    };
    let importer = registry
        .get_importer(&format_id)
        .ok_or_else(|| anyhow::anyhow!("no importer registered for format '{}'", format_id))?;
    let dataset_name = name.ok_or_else(|| anyhow::anyhow!("--name is required when importing"))?;
    let dataset = importer
        .import(catalog.db(), &storage, path, dataset_name)
        .with_context(|| format!("failed to import '{}' as {}", path.display(), format_id))?;

    println!(
        "{} Imported '{}' as dataset '{}' (id={}, format={})",
        "✓".green().bold(),
        path.display().to_string().cyan(),
        dataset.name.cyan(),
        dataset.id,
        dataset.format
    );
    Ok(())
}

fn cmd_export(registry: &FormatRegistry, name: &str, output: &Path, format: &str) -> Result<()> {
    let catalog = open_catalog()?;
    let storage = StorageManager::new(catalog.data_path());
    let dataset = DatasetService::get(catalog.db(), name)
        .with_context(|| format!("failed to find dataset '{name}'"))?;
    let format_id = DatasetFormat::new(format).to_string();
    let exporter = registry
        .get_exporter(&format_id)
        .ok_or_else(|| anyhow::anyhow!("no exporter registered for format '{}'", format_id))?;
    exporter
        .export(catalog.db(), &storage, &dataset, output)
        .with_context(|| format!("failed to export dataset '{name}' as {}", format_id))?;

    println!(
        "{} Exported dataset '{}' to {} as {}",
        "✓".green().bold(),
        dataset.name.cyan(),
        output.display().to_string().cyan(),
        format_id
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

fn cmd_label_studio_import(url: &str, api_key: &str, project: i64, name: &str) -> Result<()> {
    let catalog = open_catalog()?;
    let client = LabelStudioClient::new(url, api_key);
    let dataset = client
        .import_project(project, catalog.db(), name)
        .with_context(|| {
            format!("failed to import Label Studio project {project} into '{name}'")
        })?;

    println!(
        "{} Imported Label Studio project {} into dataset '{}' (id={})",
        "✓".green().bold(),
        project,
        dataset.name.cyan(),
        dataset.id
    );

    Ok(())
}

fn cmd_label_studio_export(
    url: &str,
    api_key: &str,
    project: i64,
    dataset: &str,
    port: u16,
) -> Result<()> {
    let catalog = open_catalog()?;
    let client = LabelStudioClient::new(url, api_key);
    client
        .export_to_project(catalog.db(), dataset, project, port)
        .with_context(|| {
            format!("failed to export dataset '{dataset}' to Label Studio project {project}")
        })?;

    println!(
        "{} Exported dataset '{}' to Label Studio project {} using port {}",
        "✓".green().bold(),
        dataset.cyan(),
        project,
        port
    );

    Ok(())
}

fn cmd_embed(dataset_name: &str, model: &Path, batch_size: usize) -> Result<()> {
    let catalog = open_catalog()?;
    let dataset = DatasetService::get(catalog.db(), dataset_name)
        .with_context(|| format!("failed to find dataset '{dataset_name}'"))?;

    #[cfg(feature = "python")]
    {
        let stored = dman_python::embeddings::compute_embeddings(
            catalog.db(),
            dataset.id,
            model,
            batch_size,
        )
        .with_context(|| {
            format!(
                "failed to compute embeddings for dataset '{}' using {}",
                dataset_name,
                model.display()
            )
        })?;

        println!(
            "{} Stored {} embeddings for dataset '{}' using {}",
            "✓".green().bold(),
            stored,
            dataset.name.cyan(),
            model.display().to_string().cyan()
        );

        Ok(())
    }

    #[cfg(not(feature = "python"))]
    {
        let _ = dataset;
        let _ = model;
        let _ = batch_size;
        anyhow::bail!("python support is not enabled for this build")
    }
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

fn build_registry() -> FormatRegistry {
    let mut registry = FormatRegistry::default_registry();

    #[cfg(feature = "python")]
    {
        if let Ok(catalog) = Catalog::open()
            && let Ok(plugin_registry) =
                dman_python::plugins::format::load_python_format_registry(vec![
                    catalog.plugins_path(),
                ])
        {
            let (importers, exporters) = plugin_registry.into_parts();
            for importer in importers {
                registry.register_importer(importer);
            }
            for exporter in exporters {
                registry.register_exporter(exporter);
            }
        }
    }

    registry
}
