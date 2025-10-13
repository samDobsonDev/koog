package ai.koog.rag.base.files

import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.text.PDFTextStripper
import java.io.File

/**
 * A [DocumentProvider] for PDF files using Apache PDFBox.
 *
 * Loads a PDF from a [File] and extracts its text using [PDFTextStripper].
 * Closes the PDF document automatically after extracting the text.
 */
public object PDFBoxDocumentProvider : DocumentProvider<File, PDDocument> {

    override suspend fun document(path: File): PDDocument? {
        // Open the PDF. Caller must close it when done.
        return PDDocument.load(path)
    }

    override suspend fun text(document: PDDocument): CharSequence {
        // Extract all text (you can customize for specific pages if desired)
        val text = PDFTextStripper().getText(document)
        document.close() // Automatically close after extracting text
        return text
    }
}
