use actix_files::NamedFile;
use actix_multipart::Multipart;
use actix_web::{web, App, HttpResponse, HttpServer, Responder, Result};
use futures_util::StreamExt;
use std::path::PathBuf;
use std::{env, fs};
use uuid::Uuid;
use chrono::DateTime;
use sanitize_filename::sanitize;
use std::process::Command;
use std::io::Write;

/// Simple index HTML
static INDEX_HTML: &str = r#"<!doctype html>
<title>Local KG Builder</title>
<h2 style="font-family:Arial">Local Knowledge Graph Builder</h2>
<form method=post enctype=multipart/form-data action="/upload">
  <label>Upload a document (.txt .pdf .docx):</label><br>
  <input type=file name=file required>
  <input type=submit value="Build KG">
</form>
<p>
<a href="/graphs">View saved graphs</a>
</p>
<hr>
<p style="color:gray;font-size:0.9em">Files saved to: {OUT_DIR}</p>
"#;

/// Small list HTML pieces
static LIST_HTML_HEAD: &str = r#"<!doctype html>
<title>Saved Graphs</title>
<h2 style="font-family:Arial">Saved Graphs</h2>
<p><a href="/">← Upload another document</a></p>
"#;

static LIST_HTML_TAIL: &str = r#"<hr>
<p style="color:gray;font-size:0.9em">Temporary files folder: {OUT_DIR}</p>
"#;

/// Create/return app directories (tmp_graphs, uploads, html)
fn app_dirs() -> (PathBuf, PathBuf, PathBuf) {
    let appdir = env::current_dir().expect("cannot determine cwd");
    let out_dir = appdir.join("tmp_graphs");
    let upload_dir = out_dir.join("uploads");
    let html_dir = out_dir.join("html");
    fs::create_dir_all(&upload_dir).ok();
    fs::create_dir_all(&html_dir).ok();
    (out_dir, upload_dir, html_dir)
}

/// GET /
async fn index() -> impl Responder {
    let (out_dir, _u, _h) = app_dirs();
    let html = INDEX_HTML.replace("{OUT_DIR}", &out_dir.to_string_lossy());
    HttpResponse::Ok().content_type("text/html").body(html)
}

/// POST /upload
/// Accepts multipart file field named `file`, saves it, calls py_helper.py, returns HTML.
async fn upload(mut payload: Multipart) -> Result<HttpResponse> {
    let (_out_dir, upload_dir, html_dir) = app_dirs();

    while let Some(field_res) = payload.next().await {
        let mut field = field_res?;
        let cd = field.content_disposition().clone();
        let filename = cd
            .get_filename()
            .map(|s| sanitize(s))
            .unwrap_or_else(|| "upload.bin".to_string());

        let uid = Uuid::new_v4().to_string();
        let saved_name = format!("{}_{}", uid, filename);
        let filepath = upload_dir.join(&saved_name);
        let filepath_clone = filepath.clone();

        // Stream chunks to file by opening for append for every chunk (simple & safe).
        while let Some(chunk_res) = field.next().await {
            let chunk = chunk_res?;
            let data_vec = chunk.to_vec();
            let filepath_for_block = filepath_clone.clone();

            // Use blocking thread to write chunk (OpenOptions append creates file automatically)
            web::block(move || -> std::io::Result<()> {
                let mut f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&filepath_for_block)?;
                f.write_all(&data_vec)?;
                Ok(())
            }).await??;
        }

        // Call Python helper (py_helper.py <input_path> <output_html_path> <orig_name>)
        let out_html = html_dir.join(format!("{}.html", uid));
        let python_path = if cfg!(target_os = "windows") {
            if std::path::Path::new("./.venv/Scripts/python.exe").exists() {
                "./.venv/Scripts/python.exe"
            } else {
                "./.venv/Scripts/python3.exe"
            }
        } else {
            "./.venv/bin/python"
        };

        let py_cmd = Command::new(python_path)
            .arg("py_helper.py")
            .arg(filepath.to_string_lossy().to_string())
            .arg(out_html.to_string_lossy().to_string())
            .arg(&filename)
            .output();

        let py_out = match py_cmd {
            Ok(o) => o,
            Err(e) => {
                let msg = format!("Failed to spawn python helper: {}", e);
                return Ok(HttpResponse::InternalServerError().body(msg));
            }
        };

        if !py_out.status.success() {
            let stderr = String::from_utf8_lossy(&py_out.stderr);
            let msg = format!("Python helper failed:\n{}", stderr);
            return Ok(HttpResponse::InternalServerError().body(msg));
        }

        let html_out = String::from_utf8_lossy(&py_out.stdout).to_string();
        return Ok(HttpResponse::Ok().content_type("text/html").body(html_out));
    }

    Ok(HttpResponse::BadRequest().body("No file field in multipart request"))
}

/// GET /graphs
async fn list_graphs() -> Result<HttpResponse> {
    let (out_dir, upload_dir, html_dir) = app_dirs();
    let mut items_html = String::new();

    let mut entries: Vec<_> = match fs::read_dir(&html_dir) {
        Ok(rd) => rd.filter_map(|r| r.ok()).collect(),
        Err(_) => Vec::new(),
    };

    // sort by modified time desc
    entries.sort_by_key(|d| {
        std::cmp::Reverse(
            d.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
        )
    });

    if entries.is_empty() {
        items_html.push_str("<p>No saved graphs yet.</p>");
    } else {
        items_html.push_str("<ul>\n");
        for e in entries {
            let path = e.path();
            if path.extension().and_then(|s| s.to_str()) != Some("html") {
                continue;
            }
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

            // ---------- REPLACED safe `created` computation ----------
            let created = e
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| {
                    t.duration_since(std::time::UNIX_EPOCH).ok().map(|d| {
                        let secs = d.as_secs() as i64;
                        DateTime::from_timestamp(secs, 0)
                            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                            .unwrap_or_else(|| "unknown".to_string())
                    })
                })
                .unwrap_or_else(|| "unknown".to_string());
            // ---------------------------------------------------------

            let uploaded = fs::read_dir(&upload_dir)
                .unwrap_or_else(|_| fs::read_dir(".").unwrap())
                .filter_map(|r| r.ok())
                .find_map(|p| {
                    let name = p.file_name().into_string().ok()?;
                    if name.starts_with(stem) { Some(name) } else { None }
                });

            items_html.push_str(&format!(
                "<li><strong>{}</strong> &nbsp;[uploaded: {}] &nbsp;[created: {}] &nbsp;<a href=\"/view/{}\" target=\"_blank\">Open</a> &nbsp; <a href=\"/uploaded/{}\" target=\"_blank\">Download Upload</a> &nbsp; <a href=\"/delete/{}\" onclick=\"return confirm('Delete this graph and uploaded file?')\">Delete</a></li>\n",
                stem,
                uploaded.as_deref().unwrap_or("(none)"),
                created,
                path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
                uploaded.as_deref().unwrap_or(""),
                stem
            ));
        }
        items_html.push_str("</ul>\n");
    }

    let full = format!("{}{}{}", LIST_HTML_HEAD, items_html, LIST_HTML_TAIL.replace("{OUT_DIR}", &out_dir.to_string_lossy()));
    Ok(HttpResponse::Ok().content_type("text/html").body(full))
}

/// GET /view/{filename}
async fn view_html(path: web::Path<String>) -> Result<NamedFile> {
    let (_out_dir, _u, html_dir) = app_dirs();
    let filename = path.into_inner();
    let p = html_dir.join(filename);
    let file = NamedFile::open(p).map_err(|e| actix_web::error::ErrorNotFound(e))?;
    Ok(file)
}

/// GET /uploaded/{filename}
async fn download_uploaded(path: web::Path<String>) -> Result<NamedFile> {
    let (_out_dir, upload_dir, _h) = app_dirs();
    let filename = path.into_inner();
    let p = upload_dir.join(filename);
    let file = NamedFile::open(p).map_err(|e| actix_web::error::ErrorNotFound(e))?;
    Ok(file)
}

/// GET /delete/{id}
async fn delete_graph(path: web::Path<String>) -> Result<HttpResponse> {
    let id = path.into_inner();
    let (_out_dir, upload_dir, html_dir) = app_dirs();
    let htmlf = html_dir.join(format!("{}.html", id));
    if htmlf.exists() {
        let _ = fs::remove_file(&htmlf);
    }
    for entry in fs::read_dir(&upload_dir).unwrap_or_else(|_| fs::read_dir(".").unwrap()).filter_map(|r| r.ok()) {
        let name = entry.file_name().into_string().ok().unwrap_or_default();
        if name.starts_with(&id) {
            let _ = fs::remove_file(entry.path());
        }
    }
    Ok(HttpResponse::SeeOther().insert_header(("Location", "/graphs")).finish())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Starting local KG web app. Open http://127.0.0.1:5000 in your browser.");
    HttpServer::new(|| {
        App::new()
            .route("/", web::get().to(index))
            .route("/upload", web::post().to(upload))
            .route("/graphs", web::get().to(list_graphs))
            .route("/view/{filename}", web::get().to(view_html))
            .route("/uploaded/{filename}", web::get().to(download_uploaded))
            .route("/delete/{id}", web::get().to(delete_graph))
    })
    .bind(("127.0.0.1", 5000))?
    .run()
    .await
}
