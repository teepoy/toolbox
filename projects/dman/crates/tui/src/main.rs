use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use dman_core::{dataset::DatasetService, db::Database, types::Dataset};
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Tabs},
    Frame,
};

#[derive(Debug, Clone, PartialEq)]
enum View {
    List,
    Detail {
        dataset_name: String,
        tab: usize,
        scroll: usize,
    },
}

#[derive(Debug, Clone)]
struct DetailData {
    image_count: u64,
    annotation_count: u64,
    categories: Vec<String>,
    images: Vec<(String, u64)>,
    schema_text: Option<String>,
}

struct App {
    datasets: Vec<Dataset>,
    selected: usize,
    should_quit: bool,
    view: View,
    detail_data: Option<DetailData>,
}

impl App {
    fn new(datasets: Vec<Dataset>) -> Self {
        Self {
            datasets,
            selected: 0,
            should_quit: false,
            view: View::List,
            detail_data: None,
        }
    }

    fn load_detail_data(&mut self) {
        if let View::Detail { dataset_name, .. } = &self.view {
            let name = dataset_name.clone();
            self.detail_data = load_detail(&name);
        }
    }
}

fn handle_key(app: &mut App, key: KeyEvent) {
    match &app.view.clone() {
        View::List => handle_key_list(app, key),
        View::Detail { tab, scroll, .. } => handle_key_detail(app, key, *tab, *scroll),
    }
}

fn handle_key_list(app: &mut App, key: KeyEvent) {
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
        KeyCode::Enter => {
            if !app.datasets.is_empty() {
                let name = app.datasets[app.selected].name.clone();
                app.view = View::Detail {
                    dataset_name: name,
                    tab: 0,
                    scroll: 0,
                };
                app.detail_data = None;
                app.load_detail_data();
            }
        }
        _ => {}
    }
}

fn handle_key_detail(app: &mut App, key: KeyEvent, current_tab: usize, current_scroll: usize) {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('r') {
        app.detail_data = None;
        app.load_detail_data();
        return;
    }

    match key.code {
        KeyCode::Esc | KeyCode::Backspace => {
            app.view = View::List;
            app.detail_data = None;
        }
        KeyCode::Char('q') | KeyCode::Char('Q') => {
            app.should_quit = true;
        }
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.should_quit = true;
        }
        KeyCode::Tab => {
            if let View::Detail {
                dataset_name,
                scroll,
                ..
            } = &app.view
            {
                let new_tab = (current_tab + 1) % 4;
                app.view = View::Detail {
                    dataset_name: dataset_name.clone(),
                    tab: new_tab,
                    scroll: *scroll,
                };
            }
        }
        KeyCode::Char(c) if ('1'..='4').contains(&c) => {
            let idx = (c as usize) - ('1' as usize);
            if let View::Detail {
                dataset_name,
                scroll,
                ..
            } = &app.view
            {
                app.view = View::Detail {
                    dataset_name: dataset_name.clone(),
                    tab: idx,
                    scroll: *scroll,
                };
            }
        }
        KeyCode::Char('j') | KeyCode::Down => {
            if current_tab == 1 {
                let max_scroll = app
                    .detail_data
                    .as_ref()
                    .map(|d| d.images.len().saturating_sub(1))
                    .unwrap_or(0);
                if let View::Detail {
                    dataset_name, tab, ..
                } = &app.view
                {
                    app.view = View::Detail {
                        dataset_name: dataset_name.clone(),
                        tab: *tab,
                        scroll: (current_scroll + 1).min(max_scroll),
                    };
                }
            }
        }
        KeyCode::Char('k') | KeyCode::Up => {
            if current_tab == 1
                && let View::Detail {
                    dataset_name, tab, ..
                } = &app.view
            {
                app.view = View::Detail {
                    dataset_name: dataset_name.clone(),
                    tab: *tab,
                    scroll: current_scroll.saturating_sub(1),
                };
            }
        }
        _ => {}
    }
}

fn ui(f: &mut Frame, app: &App) {
    match &app.view {
        View::List => draw_list(f, app),
        View::Detail {
            dataset_name,
            tab,
            scroll,
        } => draw_detail(f, app, dataset_name, *tab, *scroll),
    }
}

fn draw_list(f: &mut Frame, app: &App) {
    let area = f.area();

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
        Span::raw("Detail"),
    ]))
    .block(Block::new().borders(Borders::ALL));
    f.render_widget(status, outer_chunks[2]);
}

fn draw_detail(f: &mut Frame, app: &App, dataset_name: &str, tab: usize, scroll: usize) {
    let area = f.area();

    let dataset = app.datasets.iter().find(|d| d.name == dataset_name);

    let outer_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
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
        Span::raw(" — "),
        Span::styled(
            dataset_name,
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
    ]))
    .block(Block::new().borders(Borders::ALL));
    f.render_widget(title, outer_chunks[0]);

    let tab_titles = vec!["1:Info", "2:Images", "3:Categories", "4:Schema"];
    let tabs = Tabs::new(tab_titles)
        .select(tab)
        .style(Style::default().fg(Color::DarkGray))
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .block(Block::new().borders(Borders::ALL));
    f.render_widget(tabs, outer_chunks[1]);

    match tab {
        0 => draw_detail_info(f, app, dataset, outer_chunks[2]),
        1 => draw_detail_images(f, app, outer_chunks[2], scroll),
        2 => draw_detail_categories(f, app, outer_chunks[2]),
        3 => draw_detail_schema(f, app, dataset, outer_chunks[2]),
        _ => {}
    }

    let status = Paragraph::new(Line::from(vec![
        Span::styled(
            " [Esc] ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("Back  "),
        Span::styled(
            "[Tab/1-4] ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("Switch Tab  "),
        Span::styled(
            "[↑/↓] ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("Scroll (Images)  "),
        Span::styled(
            "[Ctrl+R] ",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("Refresh"),
    ]))
    .block(Block::new().borders(Borders::ALL));
    f.render_widget(status, outer_chunks[3]);
}

fn draw_detail_info(
    f: &mut Frame,
    app: &App,
    dataset: Option<&Dataset>,
    area: ratatui::layout::Rect,
) {
    let lines = if let Some(ds) = dataset {
        let format_str = match &ds.format {
            dman_core::types::DatasetFormat::Yolo => "YOLO".to_string(),
            dman_core::types::DatasetFormat::Coco => "COCO".to_string(),
            dman_core::types::DatasetFormat::HuggingFace => "HuggingFace".to_string(),
            dman_core::types::DatasetFormat::Custom(s) => s.clone(),
        };

        let (image_count_str, annotation_count_str) = if let Some(ref data) = app.detail_data {
            (
                data.image_count.to_string(),
                data.annotation_count.to_string(),
            )
        } else {
            ("loading...".to_string(), "loading...".to_string())
        };

        vec![
            Line::from(vec![
                Span::styled("  Name:         ", Style::default().fg(Color::Cyan)),
                Span::raw(ds.name.clone()),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Path:         ", Style::default().fg(Color::Cyan)),
                Span::raw(ds.path.to_string_lossy().to_string()),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Format:       ", Style::default().fg(Color::Cyan)),
                Span::raw(format_str),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Images:       ", Style::default().fg(Color::Cyan)),
                Span::raw(image_count_str),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Annotations:  ", Style::default().fg(Color::Cyan)),
                Span::raw(annotation_count_str),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Created:      ", Style::default().fg(Color::Cyan)),
                Span::raw(ds.created_at.clone()),
            ]),
        ]
    } else {
        vec![Line::from(Span::styled(
            "Dataset not found",
            Style::default().fg(Color::Red),
        ))]
    };

    let paragraph = Paragraph::new(lines)
        .block(
            Block::new()
                .title(" Info ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green)),
        )
        .wrap(ratatui::widgets::Wrap { trim: false });
    f.render_widget(paragraph, area);
}

fn draw_detail_images(f: &mut Frame, app: &App, area: ratatui::layout::Rect, scroll: usize) {
    let items: Vec<ListItem> = if let Some(ref data) = app.detail_data {
        if data.images.is_empty() {
            vec![ListItem::new(Span::styled(
                " No images in this dataset.",
                Style::default().fg(Color::DarkGray),
            ))]
        } else {
            data.images
                .iter()
                .skip(scroll)
                .map(|(fname, ann_count)| {
                    ListItem::new(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(fname, Style::default().add_modifier(Modifier::BOLD)),
                        Span::raw("  "),
                        Span::styled(
                            format!("[{} ann]", ann_count),
                            Style::default().fg(Color::Yellow),
                        ),
                    ]))
                })
                .collect()
        }
    } else {
        vec![ListItem::new(Span::styled(
            " Loading images...",
            Style::default().fg(Color::DarkGray),
        ))]
    };

    let list = List::new(items).block(
        Block::new()
            .title(" Images ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Blue)),
    );
    f.render_widget(list, area);
}

fn draw_detail_categories(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let items: Vec<ListItem> = if let Some(ref data) = app.detail_data {
        if data.categories.is_empty() {
            vec![ListItem::new(Span::styled(
                " No categories defined.",
                Style::default().fg(Color::DarkGray),
            ))]
        } else {
            data.categories
                .iter()
                .map(|cat| {
                    ListItem::new(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(cat, Style::default().add_modifier(Modifier::BOLD)),
                    ]))
                })
                .collect()
        }
    } else {
        vec![ListItem::new(Span::styled(
            " Loading categories...",
            Style::default().fg(Color::DarkGray),
        ))]
    };

    let list = List::new(items).block(
        Block::new()
            .title(" Categories ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Magenta)),
    );
    f.render_widget(list, area);
}

fn draw_detail_schema(
    f: &mut Frame,
    app: &App,
    dataset: Option<&Dataset>,
    area: ratatui::layout::Rect,
) {
    let content = if let Some(ref data) = app.detail_data {
        if let Some(ref schema_text) = data.schema_text {
            schema_text.clone()
        } else {
            "No schema defined".to_string()
        }
    } else if let Some(ds) = dataset {
        if ds.schema_path.is_some() {
            "Loading schema...".to_string()
        } else {
            "No schema defined".to_string()
        }
    } else {
        "No schema defined".to_string()
    };

    let paragraph = Paragraph::new(content)
        .block(
            Block::new()
                .title(" Schema ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow)),
        )
        .wrap(ratatui::widgets::Wrap { trim: false });
    f.render_widget(paragraph, area);
}

fn open_catalog_db() -> Option<Database> {
    let home = std::env::var("DMAN_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| dirs::home_dir().unwrap_or_default().join(".dman"));
    let catalog = home.join("catalog.db");
    if !catalog.exists() {
        return None;
    }
    Database::open(&catalog).ok()
}

fn load_datasets() -> Vec<Dataset> {
    let db = match open_catalog_db() {
        Some(db) => db,
        None => return vec![],
    };
    DatasetService::list(&db).unwrap_or_default()
}

fn load_detail(dataset_name: &str) -> Option<DetailData> {
    let db = open_catalog_db()?;

    let info = DatasetService::inspect(&db, dataset_name).ok()?;
    let dataset_id = info.dataset.id;

    let images: Vec<(String, u64)> = {
        let mut stmt = db
            .conn
            .prepare(
                "SELECT i.file_name, COUNT(a.id) as ann_count \
                 FROM images i \
                 LEFT JOIN annotations a ON a.image_id = i.id \
                 WHERE i.dataset_id = ?1 \
                 GROUP BY i.id \
                 ORDER BY i.file_name",
            )
            .ok()?;
        stmt.query_map(rusqlite::params![dataset_id], |row| {
            let fname: String = row.get(0)?;
            let cnt: i64 = row.get(1)?;
            Ok((fname, cnt as u64))
        })
        .ok()?
        .filter_map(|r| r.ok())
        .collect()
    };

    let schema_text = info
        .dataset
        .schema_path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok());

    let categories: Vec<String> = info.categories.iter().map(|c| c.name.clone()).collect();

    Some(DetailData {
        image_count: info.image_count,
        annotation_count: info.annotation_count,
        categories,
        images,
        schema_text,
    })
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

        if crossterm::event::poll(std::time::Duration::from_millis(100))?
            && let crossterm::event::Event::Key(key) = crossterm::event::read()?
        {
            handle_key(app, key);
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

    fn ctrl_key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::CONTROL)
    }

    #[test]
    fn test_app_initial_state() {
        let app = App::new(vec![]);
        assert_eq!(app.selected, 0);
        assert!(!app.should_quit);
        assert!(app.datasets.is_empty());
        assert_eq!(app.view, View::List);
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
        handle_key(&mut app, key(KeyCode::Char('j')));
        handle_key(&mut app, key(KeyCode::Char('k')));
        assert_eq!(app.selected, 0);
    }

    #[test]
    fn test_enter_navigates_to_detail() {
        let datasets = vec![make_dataset(1, "my_dataset")];
        let mut app = App::new(datasets);

        handle_key(&mut app, key(KeyCode::Enter));

        assert!(
            matches!(&app.view, View::Detail { dataset_name, tab: 0, scroll: 0 } if dataset_name == "my_dataset"),
            "Expected Detail view with dataset_name=my_dataset, tab=0, scroll=0, got {:?}",
            app.view
        );
    }

    #[test]
    fn test_enter_on_empty_list_stays_in_list() {
        let mut app = App::new(vec![]);
        handle_key(&mut app, key(KeyCode::Enter));
        assert_eq!(
            app.view,
            View::List,
            "Enter on empty list should not change view"
        );
    }

    #[test]
    fn test_esc_returns_to_list() {
        let datasets = vec![make_dataset(1, "ds1")];
        let mut app = App::new(datasets);
        app.view = View::Detail {
            dataset_name: "ds1".to_string(),
            tab: 0,
            scroll: 0,
        };

        handle_key(&mut app, key(KeyCode::Esc));
        assert_eq!(app.view, View::List);
    }

    #[test]
    fn test_backspace_returns_to_list() {
        let datasets = vec![make_dataset(1, "ds1")];
        let mut app = App::new(datasets);
        app.view = View::Detail {
            dataset_name: "ds1".to_string(),
            tab: 2,
            scroll: 0,
        };

        handle_key(&mut app, key(KeyCode::Backspace));
        assert_eq!(app.view, View::List);
    }

    #[test]
    fn test_tab_cycles_tabs() {
        let datasets = vec![make_dataset(1, "ds1")];
        let mut app = App::new(datasets);
        app.view = View::Detail {
            dataset_name: "ds1".to_string(),
            tab: 0,
            scroll: 0,
        };

        handle_key(&mut app, key(KeyCode::Tab));
        assert!(matches!(&app.view, View::Detail { tab: 1, .. }));

        handle_key(&mut app, key(KeyCode::Tab));
        assert!(matches!(&app.view, View::Detail { tab: 2, .. }));

        handle_key(&mut app, key(KeyCode::Tab));
        assert!(matches!(&app.view, View::Detail { tab: 3, .. }));

        handle_key(&mut app, key(KeyCode::Tab));
        assert!(matches!(&app.view, View::Detail { tab: 0, .. }));
    }

    #[test]
    fn test_number_keys_switch_tabs() {
        let datasets = vec![make_dataset(1, "ds1")];
        let mut app = App::new(datasets);
        app.view = View::Detail {
            dataset_name: "ds1".to_string(),
            tab: 0,
            scroll: 0,
        };

        handle_key(&mut app, key(KeyCode::Char('3')));
        assert!(matches!(&app.view, View::Detail { tab: 2, .. }));

        handle_key(&mut app, key(KeyCode::Char('1')));
        assert!(matches!(&app.view, View::Detail { tab: 0, .. }));

        handle_key(&mut app, key(KeyCode::Char('4')));
        assert!(matches!(&app.view, View::Detail { tab: 3, .. }));

        handle_key(&mut app, key(KeyCode::Char('2')));
        assert!(matches!(&app.view, View::Detail { tab: 1, .. }));
    }

    #[test]
    fn test_scroll_in_images_tab() {
        let datasets = vec![make_dataset(1, "ds1")];
        let mut app = App::new(datasets);
        app.view = View::Detail {
            dataset_name: "ds1".to_string(),
            tab: 1,
            scroll: 0,
        };
        app.detail_data = Some(DetailData {
            image_count: 3,
            annotation_count: 5,
            categories: vec![],
            images: vec![
                ("a.jpg".to_string(), 1),
                ("b.jpg".to_string(), 2),
                ("c.jpg".to_string(), 0),
            ],
            schema_text: None,
        });

        handle_key(&mut app, key(KeyCode::Char('j')));
        assert!(matches!(&app.view, View::Detail { scroll: 1, .. }));

        handle_key(&mut app, key(KeyCode::Down));
        assert!(matches!(&app.view, View::Detail { scroll: 2, .. }));

        handle_key(&mut app, key(KeyCode::Char('j')));
        assert!(matches!(&app.view, View::Detail { scroll: 2, .. }));

        handle_key(&mut app, key(KeyCode::Char('k')));
        assert!(matches!(&app.view, View::Detail { scroll: 1, .. }));
    }

    #[test]
    fn test_scroll_only_in_images_tab() {
        let datasets = vec![make_dataset(1, "ds1")];
        let mut app = App::new(datasets);
        app.view = View::Detail {
            dataset_name: "ds1".to_string(),
            tab: 0,
            scroll: 0,
        };
        handle_key(&mut app, key(KeyCode::Char('j')));
        assert!(
            matches!(&app.view, View::Detail { scroll: 0, .. }),
            "scroll should not change in Info tab"
        );
    }

    #[test]
    fn test_ctrl_r_in_detail_clears_detail_data() {
        let datasets = vec![make_dataset(1, "ds1")];
        let mut app = App::new(datasets);
        app.view = View::Detail {
            dataset_name: "ds1".to_string(),
            tab: 0,
            scroll: 0,
        };
        app.detail_data = Some(DetailData {
            image_count: 1,
            annotation_count: 0,
            categories: vec![],
            images: vec![],
            schema_text: None,
        });

        handle_key(&mut app, ctrl_key(KeyCode::Char('r')));
        assert!(
            matches!(&app.view, View::Detail { .. }),
            "Ctrl+R should stay in Detail view"
        );
    }

    #[test]
    fn test_quit_from_detail_view() {
        let datasets = vec![make_dataset(1, "ds1")];
        let mut app = App::new(datasets);
        app.view = View::Detail {
            dataset_name: "ds1".to_string(),
            tab: 0,
            scroll: 0,
        };

        handle_key(&mut app, key(KeyCode::Char('q')));
        assert!(app.should_quit);
    }
}
