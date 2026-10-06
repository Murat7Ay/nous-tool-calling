// Copyright (c) Murat Ay. Licensed under the MIT License.

using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace NousToolCalling;

/// <summary>
/// Decodes <c>\uXXXX</c> escapes that some servers write as literal text (for example <c>G\u00fczel</c>).
/// </summary>
/// <remarks>
/// Surrogate pairs (<c>\ud83d\ude00</c>) are joined. Code points below 0x20 (<c>\u001b</c>, <c>\u0000</c>), lone
/// surrogates and escaped backslashes (<c>\\u00fc</c>) are left as written.
/// </remarks>
public static partial class UnicodeEscapeDecoder
{
    // The longest text that may still turn into an escape: a high surrogate escape plus the start of the next one.
    [GeneratedRegex(@"\\u[dD][89abAB][0-9a-fA-F]{2}(?:\\(?:u[0-9a-fA-F]{0,3})?)?$|\\(?:u[0-9a-fA-F]{0,3})?$")]
    private static partial Regex IncompleteTailRegex();

    /// <summary>
    /// Decodes all complete escapes in <paramref name="text"/>.
    /// </summary>
    public static string Decode(string text)
    {
        if (text.IndexOf("\\u", StringComparison.Ordinal) < 0)
        {
            return text;
        }

        var sb = new StringBuilder(text.Length);
        var i = 0;
        while (i < text.Length)
        {
            if (text[i] != '\\')
            {
                sb.Append(text[i++]);
                continue;
            }

            var run = 0;
            while (i + run < text.Length && text[i + run] == '\\')
            {
                run++;
            }

            // An even run is escaped backslashes; with an odd run the last backslash may start an escape.
            if (run % 2 == 0)
            {
                sb.Append('\\', run);
                i += run;
                continue;
            }

            sb.Append('\\', run - 1);
            i += run - 1;

            if (TryReadEscape(text, i, out var high))
            {
                if (char.IsHighSurrogate((char)high) && TryReadEscape(text, i + 6, out var low) && char.IsLowSurrogate((char)low))
                {
                    sb.Append((char)high).Append((char)low);
                    i += 12;
                    continue;
                }

                if (high >= 0x20 && !char.IsSurrogate((char)high))
                {
                    sb.Append((char)high);
                    i += 6;
                    continue;
                }
            }

            sb.Append('\\');
            i++;
        }

        return sb.ToString();
    }

    private static bool TryReadEscape(string text, int index, out int codePoint)
    {
        codePoint = 0;
        return index + 6 <= text.Length
            && text[index] == '\\'
            && text[index + 1] == 'u'
            && int.TryParse(text.AsSpan(index + 2, 4), NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out codePoint);
    }

    /// <summary>
    /// Splits <paramref name="text"/> into the part that can be decoded now and a tail that may be the start of an
    /// escape continued in the next chunk.
    /// </summary>
    internal static (string Ready, string Held) SplitIncompleteTail(string text)
    {
        var match = IncompleteTailRegex().Match(text);
        if (!match.Success)
        {
            return (text, string.Empty);
        }

        // Keep the whole backslash run together so the escaped-backslash check sees all of it.
        var start = match.Index;
        while (start > 0 && text[start - 1] == '\\')
        {
            start--;
        }

        return (text[..start], text[start..]);
    }
}

/// <summary>
/// Decodes escapes in a stream of text pieces, holding back a piece's tail while it may be the start of an escape.
/// </summary>
internal sealed class StreamingUnicodeEscapeDecoder
{
    private string _held = string.Empty;

    public string Push(string piece)
    {
        var (ready, held) = UnicodeEscapeDecoder.SplitIncompleteTail(_held + piece);
        _held = held;
        return UnicodeEscapeDecoder.Decode(ready);
    }

    /// <summary>Returns whatever is still held, decoded where possible and otherwise as written.</summary>
    public string Flush()
    {
        var rest = UnicodeEscapeDecoder.Decode(_held);
        _held = string.Empty;
        return rest;
    }
}
