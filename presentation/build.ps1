# Build an editable five-slide presentation with installed Microsoft PowerPoint.
# The artifact-tool runtime is not provided in this Windows session.
# All visual elements below are native text and evidence tables.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$outputDir = Join-Path $projectRoot 'output'
$previewDir = Join-Path $projectRoot 'build/presentation-preview'
New-Item -ItemType Directory -Force -Path $outputDir, $previewDir | Out-Null
$deckPath = Join-Path $outputDir 'werewolf-ai.pptx'
$pdfPath = Join-Path $outputDir 'werewolf-ai.pdf'
$powerpoint = New-Object -ComObject PowerPoint.Application
$deck = $powerpoint.Presentations.Add(0)
$deck.PageSetup.SlideWidth = 960
$deck.PageSetup.SlideHeight = 540
$navy = 2630168
$teal = 8429568
$gray = 6316128

function Add-Text($slide, $text, $x, $y, $width, $height, $size = 22, $color = $navy, $bold = $false) {
    $shape = $slide.Shapes.AddTextbox(1, $x, $y, $width, $height)
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.WordWrap = -1
    $range = $shape.TextFrame.TextRange
    $range.Text = $text
    $range.Font.Name = 'Yu Gothic'
    $range.Font.NameFarEast = 'Yu Gothic'
    $range.Font.Size = $size
    $range.Font.Color.RGB = $color
    $range.Font.Bold = if ($bold) { -1 } else { 0 }
}

function Add-Slide($title, $notes) {
    $slide = $deck.Slides.Add($deck.Slides.Count + 1, 12)
    $slide.FollowMasterBackground = 0
    $slide.Background.Fill.ForeColor.RGB = 16777215
    Add-Text $slide $title 54 36 855 83 32 $navy $true
    Add-Text $slide ([string]$slide.SlideIndex + ' / 5') 854 506 65 20 12 $gray
    $slide.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text = $notes
    return $slide
}

function Add-Table($slide, $rows, $y, $widths) {
    $rowCount = $rows.Count
    $table = $slide.Shapes.AddTable($rowCount, $widths.Count, 54, $y, 852, ($rowCount * 53)).Table
    for ($column = 1; $column -le $widths.Count; $column++) {
        $table.Columns.Item($column).Width = $widths[$column - 1]
    }
    for ($row = 1; $row -le $rowCount; $row++) {
        for ($column = 1; $column -le $widths.Count; $column++) {
            $cell = $table.Cell($row, $column).Shape
            $cell.Fill.ForeColor.RGB = if ($row -eq 1) { $navy } else { 16119285 }
            $cell.TextFrame.MarginLeft = 14
            $cell.TextFrame.MarginRight = 10
            $cell.TextFrame.VerticalAnchor = 3
            $range = $cell.TextFrame.TextRange
            $range.Text = $rows[$row - 1][$column - 1]
            $range.Font.Name = 'Yu Gothic'
            $range.Font.NameFarEast = 'Yu Gothic'
            $range.Font.Size = 20
            $range.Font.Color.RGB = if ($row -eq 1) { 16777215 } else { $navy }
        }
    }
}

try {
    $slide = Add-Slide '説明可能な人狼AI' @'
【25秒】人狼では、誰が嘘をついているかを推定するだけでなく、他者に理由を伝え、多数決につなぐ必要があります。そこで、どの観測と規則から判断したかを追える役職推定器を実装しました。対象は5人村です。確率を判断に使いつつ、説明には根拠と代替仮説を残すことが中心課題です。
出典: docs/vision-and-principles.md、docs/requirements-explainable-role-estimator.md
'@
    Add-Text $slide 'なぜ、その人を疑うのか' 54 172 850 75 42 $teal $true
    Add-Text $slide "発言には嘘がある。自分だけが知る情報もある。`n判断の根拠を、他者が検討できる形で残す。" 54 282 840 100 26
    Add-Text $slide '5人村  村人2・占い師1・狂人1・人狼1' 54 437 850 35 20 $gray

    $slide = Add-Slide '先行研究と今回の問題意識' @'
【35秒】2020年の研究では、人狼の推定精度が71.1から72.3パーセントに向上しても、提案エージェントの勝率は46.7から45.3パーセントへ下がりました。この比較だけで因果関係は言えませんが、推定と戦略を別々に評価する必要があります。2024年には不確実な信念をBDI論理と可能世界で扱う研究もあります。今回はその方向を参考に、完全なBDI実装ではなく、共同役職分布と説明の基盤に絞りました。
一次資料: 早稲田凌亮・中原航大・渕田孝康 (2020)「深層学習による役職推定を用いた人狼知能エージェントの研究」表2・表3。https://www.jstage.jst.go.jp/article/jceeek/2020/0/2020_228/_pdf/-char/ja
一次資料: Takada and Toda (2024), https://www.jstage.jst.go.jp/article/pjsai/JSAI2024/0/JSAI2024_4Xin285/_article/-char/ja/
背景: 片上ほか (2018)「人狼知能研究のすすめ」https://www.jstage.jst.go.jp/article/jsoft/30/5/30_236/_article/-char/ja/
'@
    Add-Text $slide '推定精度の改善だけでは、勝率改善を保証できない' 54 128 852 52 25 $teal $true
    Add-Table $slide @(@('2020年研究の比較', '従来モデル', '提案モデル'), @('人狼推定精度', '71.1%', '72.3%'), @('提案エージェント勝率', '46.7%', '45.3%')) 204 @(450, 201, 201)
    Add-Text $slide '2024年の論理・可能世界研究を参考に、役職推定と説明を接続' 54 397 850 70 22
    Add-Text $slide '出典: 早稲田ほか（2020）、Takada and Toda（2024）' 54 474 840 24 14 $gray

    $slide = Add-Slide '役職の組合せを、世界全体で推定' @'
【45秒】独立に各人の人狼確率を出すと、人数の制約が崩れます。今回は村人の自己視点で24通り、公開視点で60通りの役職割当てを列挙し、その重みから確率を求めます。自分の占い結果は矛盾する世界を除外しますが、他者の占い報告は嘘の可能性を残して重みだけ変えます。更新は対数空間で行い、全員の人狼確率の合計を1に保ちます。生存候補の得点を比較し、同点は固定順で投票を決めます。
実装: belief_model.py / ExplainableRoleEstimator、TransparentLikelihoodModel
24 = 4!、60 = 5!/2!。自己が村人の場合。占い師の自己視点は12通り。
'@
    Add-Text $slide '24 世界' 54 131 385 70 44 $teal $true
    Add-Text $slide '60 世界' 504 131 385 70 44 $teal $true
    Add-Text $slide '村人の自己視点' 54 208 385 36 23
    Add-Text $slide '公開情報だけの視点' 504 208 385 36 23
    Add-Table $slide @(@('入力', '計算上の扱い'), @('自己占い・確認された襲撃死', 'ハード制約で世界を除外'), @('他者のCO・占い報告・投票', 'ソフト尤度で重みを更新')) 274 @(440, 412)
    Add-Text $slide '役職人数を保存する共同分布。投票は得点最大、同点は固定順。' 54 462 850 40 20

    $slide = Add-Slide '実装で見つけた問題と設計判断' @'
【40秒】既存コードを検査すると、私有情報を文章から削っても、確率や代替候補の順番に漏れる問題がありました。そこで公開説明を別の60世界で再計算しました。また、自分の投票を証拠として取り込むと、自分の判断を自分で強化してしまうため、自己行動を中立化しました。説明の根拠はイベントと規則へ接続し、観測への反論は、その証拠を除く再生で検討します。ソフト尤度のゼロや矛盾するIDも、黙って受け入れず拒否します。
実装中に確認した問題と修正: docs/implementation-completion.md
推論グラフ: belief_explanations.py。詳細な世界別尤度は BeliefUpdate.world_evidence。
'@
    Add-Table $slide @(@('見つかった問題', '採用した対策'), @('説明の数値から私有結果が漏れる', '公開履歴で分布ごと再計算'), @('自分の宣言で確信を増幅する', '自己行動を追加証拠にしない'), @('説明の根拠を反論・検査できない', '証拠ID・推論グラフ・除外再生')) 148 @(463, 389)
    Add-Text $slide '40 テスト' 54 401 345 70 42 $teal $true
    Add-Text $slide "同一履歴100回の再生一致`n公開説明全体の非漏洩も検査" 406 403 490 80 22

    $slide = Add-Slide '動作例と、これから検証すること' @'
【35秒】最後に合成局面を再生した動作例です。Agent02の占い師COとAgent03への黒報告、Agent04の遅い対抗COを入力しました。ハード制約のみでは固定順で02を選び、ソフト証拠を使うと03を選びます。自己以外の4人で計算したBrier scoreとlog lossは表の通りです。ただし、これは仕組みの動作例で、勝率の実証ではありません。今後は実対戦ログで尤度を検証し、推定の較正と勝率、説明が投票を動かす効果を分けて測ります。
再現: python belief_evaluation.py examples/five_player.json --output output/evaluation.json
合成局面1件、評価対象4人。真役職は採点にのみ使用。Brierは4役職の二乗誤差和を対象人数で平均（範囲0〜2）。log lossは自然対数、下限1e-15。測定値は丸めて表示。
本実装の対象外: 自由文生成、実戦勝率の測定、15人近似推論、学習、一貫した嘘生成。
'@
    Add-Text $slide '合成局面1件の再生結果（値が小さいほど良い）' 54 132 850 40 23
    Add-Table $slide @(@('条件', 'Brier score', 'log loss', '投票'), @('ハード制約のみ', '0.750', '1.386', '02'), @('ソフト証拠あり', '0.377', '0.694', '03')) 204 @(350, 185, 185, 132)
    Add-Text $slide '実戦勝率・説得効果は未検証' 54 394 850 47 29 $teal $true
    Add-Text $slide '次は実対戦ログで尤度を検証し、推定と行動を別々に評価する' 54 457 852 45 20

    $deck.SaveAs($deckPath, 24)
    $deck.Export($previewDir, 'PNG', 1600, 900)
    $deck.SaveAs($pdfPath, 32)
    # Native layout check against PowerPoint's own text measurement.
    $issues = @()
    foreach ($slide in $deck.Slides) {
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
                if ($shape.TextFrame.TextRange.BoundHeight -gt ($shape.Height + 3)) {
                    $issues += "Slide $($slide.SlideIndex): overflow $($shape.Name)"
                }
            }
        }
    }
    $issues | ConvertTo-Json | Set-Content (Join-Path $previewDir 'layout-issues.json') -Encoding utf8
    Write-Output "Created $deckPath, $pdfPath; slides=$($deck.Slides.Count); layoutIssues=$($issues.Count)"
} finally {
    $deck.Close()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($deck)
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerpoint)
}
