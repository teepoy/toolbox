use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use dman_core::{dataset::DatasetService, db::Database, types::Dataset};
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Frame,
};

struct App {
    datasets: Vec<Dataset>,
    selected: usize,
    should_quit: bool,
}

impl App {
    fn new(datasets: Vec<Dataset>) -> Self {
        Self {
            datasets,
            selected: 0,
            should_quit: false,
        }
    }
}

fn handle_key(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Char('q') | KeyCode::Char('Q') => {
            app.should_quit = true;
        }
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.should_quit = true;
        }
        KeyCode::Char('j') | KeyCode::Down => {
            if !app.datasets.is_empty() {
                app.selected = (app.selected + 1).min(app.datasets.len() - 1);
            }
        }
        KeyCode::Char('k') | KeyCode::Up => {
            if app.selected > 0 {
                app.selected -= 1;
            }
        }
        _ => {}
    }
}

fn ui(f: &mut Frame, app: &App) {
    let area = f.area();

    // Outer vertical split: title bar (1 line) / main area / status bar (1 line)
    let outer_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(area);

    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            "dman",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" — Dataset Manager"),
    ]))
    .block(Block::new().borders(Borders::ALL));
    f.render_widget(title, outer_chunks[0]);

    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(outer_chunks[1]);

    let items: Vec<ListItem> = if app.datasets.is_empty() {
        vec![ListItem::new(Span::styled(
            " No datasets found. Run `dman init` to create a catalog.",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        app.datasets
            .iter()
            .map(|ds| {
                let format_label = match &ds.format {
                    dman_core::types::DatasetFormat::Yolo => "YOLO",
                    dman_core::types::DatasetFormat::Coco => "COCO",
                    dman_core::types::DatasetFormat::HuggingFace => "HF",
                    dman_core::types::DatasetFormat::Custom(s) => s.as_str(),
                };
                ListItem::new(Line::from(vec![
                    Span::raw(" "),
                    Span::styled(&ds.name, Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw("  "),
                    Span::styled(
                        format!("[{}]", format_label),
                        Style::default().fg(Color::Yellow),
                    ),
                ]))
            })
            .collect()
    };

    let mut list_state = ListState::default();
    if !app.datasets.is_empty() {
        list_state.select(Some(app.selected));
    }

    let list = List::new(items)
        .block(
            Block::new()
                .title(" Datasets ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue)),
        )
        .highlight_style(
            Style::default()
                .bg(Color::Blue)
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");

    f.render_stateful_widget(list, main_chunks[0], &mut list_state);

    let detail_text = if app.datasets.is_empty() {
        vec![Line::from(Span::styled(
            "No dataset selected",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        let ds = &app.datasets[app.selected];
        let format_str = match &ds.format {
            dman_core::types::DatasetFormat::Yolo => "YOLO".to_string(),
            dman_core::types::DatasetFormat::Coco => "COCO".to_string(),
            dman_core::types::DatasetFormat::HuggingFace => "HuggingFace".to_string(),
            dman_core::types::DatasetFormat::Custom(s) => s.clone(),
        };
        vec![
            Line::from(vec![
                Span::styled("Name: ", Style::default().fg(Color::Cyan)),
                Span::raw(ds.name.clone()),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Format: ", Style::default().fg(Color::Cyan)),
                Span::raw(format_str),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Path: ", Style::default().fg(Color::Cyan)),
                Span::raw(ds.path.to_string_lossy().to_string()),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Created: ", Style::default().fg(Color::Cyan)),
                Span::raw(ds.created_at.clone()),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("ID: ", Style::default().fg(Color::Cyan)),
                Span::raw(ds.id.to_string()),
            ]),
        ]
    };

    let detail = Paragraph::new(detail_text)
        .block(
            Block::new()
                .title(" Details ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green)),
        )
        .wrap(ratatui::widgets::Wrap { trim: true });
    f.render_widget(detail, main_chunks[1]);

    let status = Paragraph::new(Line::from(vec![
        Span::styled(
            " [q] ",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ),
        Span::raw("Quit  "),
        Span::styled(
            "[↑/k] ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("Up  "),
        Span::styled(
            "[↓/j] ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("Down  "),
        Span::styled(
            "[Enter] ",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("Select"),
    ]))
    .block(Block::new().borders(Borders::ALL));
    f.render_widget(status, outer_chunks[2]);
}

fn load_datasets() -> Vec<Dataset> {
    let home = std::env::var("DMAN_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| dirs::home_dir().unwrap_or_default().join(".dman"));
    let catalog = home.join("catalog.db");
    if !catalog.exists() {
        return vec![];
    }
    let Ok(db) = Database::open(&catalog) else {
        return vec![];
    };
    DatasetService::list(&db).unwrap_or_default()
}

fn main() -> anyhow::Result<()> {
    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend)?;

    let datasets = load_datasets();
    let mut app = App::new(datasets);

    let result = run_app(&mut terminal, &mut app);

    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen
    )?;

    result
}

fn run_app(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
) -> anyhow::Result<()> {
    loop {
        terminal.draw(|f| ui(f, app))?;

        if crossterm::event::poll(std::time::Duration::from_millis(100))? {
            if let crossterm::event::Event::Key(key) = crossterm::event::read()? {
                handle_key(app, key);
            }
        }

        if app.should_quit {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use dman_core::types::{Dataset, DatasetFormat};
    use std::path::PathBuf;

    fn make_dataset(id: i64, name: &str) -> Dataset {
        Dataset {
            id,
            name: name.to_string(),
            path: PathBuf::from("/tmp/test"),
            format: DatasetFormat::Yolo,
            schema_path: None,
            created_at: "2024-01-01".to_string(),
            updated_at: None,
            metadata: None,
        }
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    #[test]
    fn test_app_initial_state() {
        let app = App::new(vec![]);
        assert_eq!(app.selected, 0);
        assert!(!app.should_quit);
        assert!(app.datasets.is_empty());
    }

    #[test]
    fn test_app_quit() {
        let mut app = App::new(vec![]);
        handle_key(&mut app, key(KeyCode::Char('q')));
        assert!(app.should_quit);
    }

    #[test]
    fn test_app_quit_uppercase() {
        let mut app = App::new(vec![]);
        handle_key(&mut app, key(KeyCode::Char('Q')));
        assert!(app.should_quit);
    }

    #[test]
    fn test_app_navigation() {
        let datasets = vec![
            make_dataset(1, "ds1"),
            make_dataset(2, "ds2"),
            make_dataset(3, "ds3"),
        ];
        let mut app = App::new(datasets);
        assert_eq!(app.selected, 0);

        handle_key(&mut app, key(KeyCode::Char('j')));
        assert_eq!(app.selected, 1);

        handle_key(&mut app, key(KeyCode::Char('j')));
        assert_eq!(app.selected, 2);
    }

    #[test]
    fn test_app_navigation_arrow_keys() {
        let datasets = vec![
            make_dataset(1, "ds1"),
            make_dataset(2, "ds2"),
            make_dataset(3, "ds3"),
        ];
        let mut app = App::new(datasets);

        handle_key(&mut app, key(KeyCode::Down));
        assert_eq!(app.selected, 1);

        handle_key(&mut app, key(KeyCode::Up));
        assert_eq!(app.selected, 0);
    }

    #[test]
    fn test_app_navigation_no_overflow_down() {
        let datasets = vec![make_dataset(1, "ds1"), make_dataset(2, "ds2")];
        let mut app = App::new(datasets);
        app.selected = 1;

        handle_key(&mut app, key(KeyCode::Char('j')));
        assert_eq!(app.selected, 1, "should not go past last item");
    }

    #[test]
    fn test_app_navigation_no_overflow_up() {
        let mut app = App::new(vec![make_dataset(1, "ds1")]);
        app.selected = 0;

        handle_key(&mut app, key(KeyCode::Char('k')));
        assert_eq!(app.selected, 0, "should not go below 0");
    }

    #[test]
    fn test_app_navigation_empty_datasets() {
        let mut app = App::new(vec![]);
        // Should not panic on j/k with empty list
        handle_key(&mut app, key(KeyCode::Char('j')));
        handle_key(&mut app, key(KeyCode::Char('k')));
        assert_eq!(app.selected, 0);
    }
}
